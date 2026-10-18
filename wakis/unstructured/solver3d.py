# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""
3D unstructured FIT on tetrahedra.

Electric voltages e on the primal edges, magnetic fluxes b on the primal
faces, magnetic voltages on the dual (Voronoi) edges and electric fluxes on
the dual faces (Clemens & Weiland 2001; Bossavit & Kettunen 1999)::

    b' = -C e
    M_eps e' = C^T M_nu b - j

with the diagonal Hodge matrices of the orthogonal Delaunay-Voronoi pair

    M_eps[e] = eps_0 sum_T eps_T A_{e,T} / L_e
    M_nu[f]  = sum_T h_{f,T} / (mu_0 mu_T) / A_f

where A_{e,T} is the part of the dual face of edge e inside tetrahedron T and
h_{f,T} the part of the dual edge of face f inside T. C is the face-edge
incidence; D C = 0 (D: cell-face incidence) so that the discrete magnetic
flux is divergence free to round-off. ``hodge="whitney"`` replaces both
matrices by the consistent Whitney-form mass matrices (non-diagonal, positive
definite on any mesh), the fallback for non-Delaunay meshes.
"""

import numpy as np
import scipy.sparse as sp
from scipy.constants import epsilon_0 as eps_0
from scipy.constants import mu_0
from scipy.sparse.linalg import LinearOperator, eigsh, splu

from ._core import (
    ETA_0,
    CapacitiveUpdate,
    Recorder,
    cell_materials,
    check_dispersive_stability,
    max_eigenvalue,
)


def whitney_mass_matrices(mesh, eps_r, mu_r, live):
    """
    Consistent Whitney 1-form (edge) and 2-form (face) mass matrices.

    M1[e, e'] = eps_0 int eps W_e . W_e' dV,
    M2[f, f'] = int (1 / mu_0 mu) w_f . w_f' dV,
    with the lowest-order Whitney functions on each tetrahedron (exact
    integration of products of barycentric coordinates,
    int lambda_i lambda_j dV = V (1 + delta_ij) / 20).
    """
    nc = mesh.n_cells
    grad = mesh.barycentric_gradients()  # (nc, 4, 3)
    V = mesh.cell_volume
    Mll = (np.ones((4, 4)) + np.eye(4)) / 20  # int lam_i lam_j / V
    rows, cols, vals = [], [], []
    loc_e = mesh._local_edges
    sgn_e = np.stack(
        [np.where(mesh.cells[:, a] < mesh.cells[:, b], 1.0, -1.0) for a, b in loc_e], 1
    )
    for p, (a, b) in enumerate(loc_e):
        for q, (c, d) in enumerate(loc_e):
            # W_ab . W_cd = l_a l_c gb.gd - l_a l_d gb.gc - l_b l_c ga.gd + l_b l_d ga.gc
            dot = lambda i, j: np.einsum("ni,ni->n", grad[:, i], grad[:, j])
            val = (
                Mll[a, c] * dot(b, d)
                - Mll[a, d] * dot(b, c)
                - Mll[b, c] * dot(a, d)
                + Mll[b, d] * dot(a, c)
            )
            rows.append(mesh.cell_edges[:, p])
            cols.append(mesh.cell_edges[:, q])
            vals.append(eps_0 * eps_r * live * V * val * sgn_e[:, p] * sgn_e[:, q])
    ne = mesh.n_edges
    M1 = sp.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(ne, ne),
    )
    # Face functions w_f = 2 (l_i gj x gk + l_j gk x gi + l_k gi x gj) for
    # the local face (i, j, k) opposite vertex m, oriented outward-consistent
    # with the global (sorted) face normal
    rows, cols, vals = [], [], []
    loc_f = mesh._local_faces
    terms = []
    for k_loc, (i, j, k) in enumerate(loc_f):
        gi, gj, gk = grad[:, i], grad[:, j], grad[:, k]
        t = [
            (i, 2 * np.cross(gj, gk)),
            (j, 2 * np.cross(gk, gi)),
            (k, 2 * np.cross(gi, gj)),
        ]
        # Global orientation: sign of the flux of w_f through the sorted face
        fid = mesh.cell_faces[:, k_loc]
        nrm = mesh.face_normal[fid]
        # w_f . n is constant on the face, sum_l t_l . n / 3 * ... use l = 1/3
        flux = (
            sum(np.einsum("ni,ni->n", v, nrm) for _, v in t) / 3 * mesh.face_area[fid]
        )
        s = np.sign(flux)
        terms.append([(l, v * s[:, None]) for l, v in t])
    for p in range(4):
        for q in range(4):
            val = np.zeros(nc)
            for lp, vp in terms[p]:
                for lq, vq in terms[q]:
                    val += Mll[lp, lq] * np.einsum("ni,ni->n", vp, vq)
            rows.append(mesh.cell_faces[:, p])
            cols.append(mesh.cell_faces[:, q])
            vals.append(live * V * val / (mu_0 * mu_r))
    nf = mesh.n_faces
    M2 = sp.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(nf, nf),
    )
    return M1, M2


class UnstructuredFIT3D:
    """
    Time- and frequency-domain FIT solver on a tetrahedral mesh.

    Parameters
    ----------
    mesh : SimplexMesh
        3D tetrahedral mesh with material tags.
    materials : dict, optional
        ``{tag: material}`` with ``[eps_r, mu_r]``, ``[eps_r, mu_r, sigma]``,
        a `wakis.dispersive.DispersiveMaterial` or ``"pec"``.
    bc : {"pec", "pmc", "abc"}
        Outer boundary: PEC (tangential E = 0), PMC (natural) or first-order
        Silver-Mueller absorbing boundary.
    hodge : {"voronoi", "whitney"}
        Diagonal circumcentric Hodge (default, explicit update) or
        consistent Whitney mass matrices (any mesh; one sparse solve per step).
    massless_tol : float, optional
        Edges whose dual face area is below ``massless_tol * L_e**2`` (nearly
        degenerate Voronoi faces) are treated as massless: their voltage is
        eliminated from the quasi-static constraint (C^T M_nu b)_e = 0 each
        step, which removes the timestep collapse they would cause.
    dt : float, optional
        Timestep [s] (default ``cfl`` x stability limit).
    cfl : float, optional
        Fraction of the stability limit 2 / sqrt(lambda_max).

    Attributes
    ----------
    e : ndarray (n_edges,)
        Edge voltages [V] (line integrals of E along the oriented edges).
    b : ndarray (n_faces,)
        Face fluxes [Wb].
    """

    def __init__(
        self,
        mesh,
        materials=None,
        bc="pec",
        hodge="voronoi",
        massless_tol=1e-2,
        dt=None,
        cfl=0.95,
    ):
        if mesh.dim != 3:
            raise ValueError("UnstructuredFIT3D needs a tetrahedral mesh")
        if bc not in ("pec", "pmc", "abc"):
            raise ValueError("bc must be 'pec', 'pmc' or 'abc'")
        if hodge not in ("voronoi", "whitney"):
            raise ValueError("hodge must be 'voronoi' or 'whitney'")
        self.mesh, self.bc, self.hodge = mesh, bc, hodge
        eps_r, mu_r, sigma, pec, dispersive = cell_materials(mesh, materials)
        self.eps_r, self.mu_r, self.sigma, self.pec_cells = eps_r, mu_r, sigma, pec
        self.dispersive = dispersive
        live = (~pec).astype(float)
        L = mesh.edge_length

        fixed = np.zeros(mesh.n_edges, bool)
        fixed[mesh.cell_edges[pec].ravel()] = True
        if bc == "pec":
            fixed[mesh.boundary_edges] = True
        touched = np.zeros(mesh.n_edges, bool)
        touched[mesh.cell_edges[~pec].ravel()] = True
        fixed |= ~touched
        free = np.flatnonzero(~fixed)

        area = mesh.dual_measure(live)
        cap = eps_0 * mesh.dual_measure(eps_r * live) / L
        cond = mesh.dual_measure(sigma * live) / L
        if bc == "abc":
            # Silver-Mueller: H_t = n x E / eta on the outer surface, i.e. a
            # conductance l_e / (eta L_e) on every boundary edge
            bcell = np.zeros(mesh.n_faces, int)
            bcell[mesh.cell_faces.ravel()] = np.repeat(np.arange(mesh.n_cells), 4)
            eta_f = ETA_0 * np.sqrt(
                mu_r[bcell[mesh.boundary_faces]] / eps_r[bcell[mesh.boundary_faces]]
            )
            eta_e = np.full(mesh.n_edges, ETA_0)
            eta_e[mesh.face_edges[mesh.boundary_faces].ravel()] = np.repeat(eta_f, 3)
            cond = cond + mesh.boundary_dual_length / (eta_e * L)
        self.Mnu_diag = mesh.dual_length(live / (mu_0 * mu_r)) / mesh.face_area

        if hodge == "voronoi":
            # Nearly degenerate Voronoi faces, also after material weighting
            # (signed per-cell parts can cancel at material interfaces)
            small = massless_tol * L**2
            massless = ((area < small) | (cap * L / eps_0 < small)) & ~fixed
            self.massless = np.flatnonzero(massless)
            self.free = np.flatnonzero(~fixed & ~massless)
        else:
            self.massless = np.array([], dtype=int)
            self.free = free
        self.fixed = np.flatnonzero(fixed)
        C = mesh.C.tocsc()
        self.Cf = C[:, self.free].tocsr()
        self.Cs = C[:, self.massless].tocsr()

        if hodge == "voronoi":
            self.Mnu = sp.diags(self.Mnu_diag)
            self.cap = cap[self.free]
            if np.any(self.cap <= 0):
                raise ValueError(
                    f"{np.sum(self.cap <= 0)} edges with non-positive dual area: "
                    "the mesh is not Delaunay; use hodge='whitney'"
                )
            Kss = (self.Cs.T @ self.Mnu @ self.Cs).tocsc()
            self._Kss = splu(Kss) if self.massless.size else None
        else:
            M1, M2 = whitney_mass_matrices(mesh, eps_r, mu_r, live)
            self.Mnu = M2.tocsr()
            M1f = M1[self.free][:, self.free].tocsc()
            self.M1 = M1f
            self._M1 = splu(M1f)
            self.cap = M1f.diagonal()
        self.cond = cond[self.free]

        # Stability limit
        Kf = (self.Cf.T @ self.Mnu @ self.Cf).tocsr()
        self.K = Kf
        if hodge == "voronoi":
            if self.massless.size:
                Ksf = (self.Cs.T @ self.Mnu @ self.Cf).tocsr()
                self._Ksf = Ksf
                minv = 1 / np.sqrt(self.cap)

                def mv(x):
                    y = minv * x
                    return minv * (Kf @ y - Ksf.T @ self._Kss.solve(Ksf @ y))

                n = self.free.size
                op = LinearOperator((n, n), matvec=mv, dtype=float)
                lam = eigsh(op, k=1, which="LA", return_eigenvectors=False, tol=1e-4)[0]
            else:
                lam = max_eigenvalue(Kf, M_inv_sqrt=1 / np.sqrt(self.cap))
        else:
            lam = max_eigenvalue(Kf, M_solve=self._M1.solve)
        self.lambda_max = lam
        self.dt_max = 2 / np.sqrt(lam)
        self.dt = cfl * self.dt_max if dt is None else float(dt)
        self.courant = self.dt * np.sqrt(lam) / 2
        self.h_mean = L.mean()

        groups = []
        if dispersive:
            if hodge != "voronoi":
                raise NotImplementedError("dispersive materials need hodge='voronoi'")
            check_dispersive_stability(
                [m for m, _ in dispersive], self.dt, self.courant
            )
            for mat, sel in dispersive:
                c = eps_0 * mesh.dual_measure(sel * live) / L
                cf = c[self.free]
                idx = np.flatnonzero(cf > 0)
                groups.append((mat, idx, cf[idx]))
        if hodge == "voronoi":
            self._update = CapacitiveUpdate(self.cap, self.cond, groups, self.dt)
        self.sources = []
        self.recorder = Recorder()
        self.reset()

    # ------------------------------------------------------------ state
    def reset(self):
        """Zero fields, auxiliary polarizations, probes and DFTs."""
        self.u = np.zeros(self.free.size)
        self.b = np.zeros(self.mesh.n_faces)
        self.n = 0
        if self.hodge == "voronoi":
            self._update.reset()
        for rec in self.recorder.probes:
            rec["samples"] = []
        for rec in self.recorder.dfts:
            rec["data"][:] = 0

    @property
    def e(self):
        """Voltages on all edges [V] (0 on PEC edges)."""
        out = np.zeros(self.mesh.n_edges)
        out[self.free] = self.u
        if self.massless.size:
            out[self.massless] = self._massless_voltages(self.u)
        return out

    def _massless_voltages(self, u):
        return -self._Kss.solve(self._Ksf @ u)

    def energy(self):
        """Electromagnetic energy [J] at the current (staggered) state."""
        if self.hodge == "voronoi":
            we = 0.5 * self.cap @ self.u**2
        else:
            we = 0.5 * self.u @ (self.M1 @ self.u)
        return we + 0.5 * self.b @ (self.Mnu @ self.b)

    def divergence_b(self):
        """Net magnetic flux out of every tetrahedron, D b [Wb]."""
        return self.mesh.D @ self.b

    # ------------------------------------------------------------ sources
    def add_point_dipole(self, position, direction, amp_func):
        """
        Electric point dipole J = I l(t) d delta(r - r0).

        Distributed on the edges of the containing tetrahedron with the
        Whitney functions, i_e = I l(t) d . W_e(r0) (Galerkin-consistent and
        independent of the mesh size).

        Parameters
        ----------
        position : (3,) float
            Location [m].
        direction : (3,) float
            Dipole direction.
        amp_func : callable
            t [s] -> current moment I l [A m].
        """
        d = np.asarray(direction, float) / np.linalg.norm(direction)
        cell, W = self.mesh.whitney_edge([position])
        w = np.zeros(self.mesh.n_edges)
        w[self.mesh.cell_edges[cell[0]]] = W[0] @ d
        self.sources.append((w[self.free], amp_func))

    def add_current_density(self, J_func, amp_func):
        """
        Distributed current density J(r) amp(t) [A/m^2].

        Edge currents i_e = A_e J(m_e) . t_e (flux through the dual face).

        Parameters
        ----------
        J_func : callable
            (n, 3) midpoints [m] -> (n, 3) current density [A/m^2].
        amp_func : callable
            t [s] -> dimensionless amplitude.
        """
        m = self.mesh
        J = np.asarray(J_func(m.edge_midpoint))
        w = m.dual_measure() * np.einsum("ij,ij->i", J, m.edge_tangent)
        self.sources.append((w[self.free], amp_func))

    # ------------------------------------------------------------ monitors
    def _whitney_interp(self, points, comp):
        cell, W = self.mesh.whitney_edge(points)
        rows = np.repeat(np.arange(len(cell)), 6)
        cols = self.mesh.cell_edges[cell].ravel()
        P = sp.csr_matrix(
            (W[:, :, comp].ravel(), (rows, cols)), shape=(len(cell), self.mesh.n_edges)
        )
        return P[:, self.free]

    def add_probe(self, points, component):
        """
        Record one Cartesian E component at points every step.

        Whitney interpolation of the free edge voltages (massless edges are
        omitted, a small error near degenerate Voronoi faces).

        Parameters
        ----------
        points : (n, 3) float
            Probe points [m].
        component : {0, 1, 2}
            x, y or z.
        """
        return self.recorder.add_probe(self._whitney_interp(points, component))

    def add_dft(self, freqs):
        """Running DFT of the free edge voltages at `freqs` [Hz]."""
        return self.recorder.add_dft(freqs, self.free.size)

    def probe_signal(self, probe):
        """Recorded samples, shape (n_steps, n_points)."""
        return np.asarray(probe["samples"])

    # ------------------------------------------------------------ stepping
    def one_step(self):
        """One leapfrog step: b to n+1/2, then e to n+1 (sources at n+1/2)."""
        dt = self.dt
        t_half = (self.n + 0.5) * dt
        src = np.zeros(self.free.size)
        for w, f in self.sources:
            src += w * f(t_half)
        e_s = self._massless_voltages(self.u) if self.massless.size else None
        self.b -= dt * (self.Cf @ self.u)
        if e_s is not None:
            self.b -= dt * (self.Cs @ e_s)
        r = self.Cf.T @ (self.Mnu @ self.b) - src
        if self.hodge == "voronoi":
            self.u = self._update.step(self.u, r)
        else:
            self.u = self.u + dt * self._M1.solve(r)
        self.n += 1
        self.recorder.record(self.u, self.n * dt, dt)

    def run(self, n_steps):
        """Advance `n_steps` steps."""
        for _ in range(int(n_steps)):
            self.one_step()

    # ------------------------------------------------------------ eigenmodes
    def eigenmodes(self, n_modes=6, f_target=None, permc_spec="COLAMD"):
        """
        Resonant frequencies of the closed cavity.

        Solves C^T M_nu C e = omega^2 M_eps e on the free edges by
        shift-invert Lanczos around `f_target` (the gradient null space at
        omega = 0 is far from the shift). Massless edges are eliminated.

        Parameters
        ----------
        n_modes : int
        f_target : float
            Shift frequency [Hz] (> 0, below the modes of interest).

        Returns
        -------
        f : ndarray
            Frequencies [Hz] of the `n_modes` modes closest to `f_target`,
            sorted ascending.
        e : ndarray (n_edges, n_modes)
            Edge voltages of the modes.
        """
        if not f_target:
            raise ValueError("f_target > 0 is required (gradient null space at 0)")
        sigma = (2 * np.pi * f_target) ** 2
        if self.hodge == "voronoi":
            Cfull = sp.hstack([self.Cf, self.Cs]).tocsr()
            K = (Cfull.T @ self.Mnu @ Cfull).tocsc()
            M = sp.diags(np.r_[self.cap, np.zeros(self.massless.size)])
            idx = np.r_[self.free, self.massless]
        else:
            K, M, idx = self.K.tocsc(), self.M1, self.free
        A = (K - sigma * M).tocsc()
        lu = splu(A, permc_spec=permc_spec)
        # Static (gradient) null space: e = G phi with phi = 0 on every node
        # touching a fixed (PEC) edge. Projecting it out M-orthogonally
        # (phi solves the discrete Poisson problem G^T M G phi = G^T M x)
        # maps those eigenvalues of the shift-invert operator to 0
        Gn = self.mesh.G.tocsr()[idx]
        touch = np.zeros(self.mesh.n_nodes, bool)
        touch[self.mesh.edges[self.fixed].ravel()] = True
        Gn = Gn[:, np.flatnonzero(~touch)].tocsc()
        MG = (M @ Gn).tocsc()
        lap = splu((Gn.T @ MG).tocsc(), permc_spec=permc_spec)

        def op(x):
            y = lu.solve(x)
            return y - Gn @ lap.solve(MG.T @ y)

        n = K.shape[0]
        OPinv = LinearOperator((n, n), matvec=op, dtype=float)
        w2, vec = eigsh(
            K,
            k=n_modes + 2,
            M=M,
            sigma=sigma,
            which="LM",
            OPinv=OPinv,
            ncv=max(4 * n_modes + 10, 40),
            tol=1e-10,
        )
        keep = np.flatnonzero(w2 > 1e-6 * sigma)
        w2, vec = w2[keep], vec[:, keep]
        # The n_modes physical modes closest to the shift
        o = np.argsort(np.abs(w2 - sigma))[:n_modes]
        o = o[np.argsort(w2[o])]
        out = np.zeros((self.mesh.n_edges, o.size))
        out[idx] = vec[:, o]
        return np.sqrt(w2[o]) / (2 * np.pi), out
