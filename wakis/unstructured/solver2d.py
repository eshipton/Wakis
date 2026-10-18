# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""
2D unstructured FIT on triangles (TM: Ez on nodes, TE: Hz on nodes).

Primal grid: the triangulation. Dual grid: its circumcentric (Voronoi) dual,
whose edges are perpendicular to the primal edges, so the material matrices
are diagonal. Per unit length in z:

TM (Ez, Hx, Hy)::

    b' = -G ez                         (magnetic flux through primal edges)
    M_eps ez' = G^T M_nu b - i_z        M_eps = eps_0 sum_T eps_T A_{n,T}
                                        M_nu  = sum_T h_{e,T} / (mu_0 mu_T L_e)

TE (Hz, Ex, Ey)::

    d' = G hz - i_e                    (electric flux through primal edges)
    M_mu hz' = -G^T v - k_z            M_mu = mu_0 sum_T mu_T A_{n,T}
                                        v_e = sum_T h_{e,T} E_{e,T}

with E_{e,T} = d_e / (L_e eps_0 eps_T) in each triangle (series: the normal
D is continuous across the edge). A_{n,T} is the part of the dual cell of
node n in triangle T and h_{e,T} the part of the dual edge of e in T. G is
the edge-node incidence. See the physics guide, "Unstructured FIT".
"""

import numpy as np
import scipy.sparse as sp
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


class UnstructuredFIT2D:
    """
    Time- and frequency-domain FIT solver on a triangular mesh.

    Parameters
    ----------
    mesh : SimplexMesh
        2D triangular mesh with material tags.
    materials : dict, optional
        ``{tag: material}``; a material is ``[eps_r, mu_r]``,
        ``[eps_r, mu_r, sigma]`` (TM only for sigma), a
        `wakis.dispersive.DispersiveMaterial` or ``"pec"``. Missing tags are
        vacuum.
    polarization : {"TM", "TE"}
        TM: Ez (E along z, tangential to every interface); TE: Hz (E in the
        plane, crossing curved interfaces).
    bc : {"pec", "pmc", "abc"}
        Outer boundary: perfect electric conductor, perfect magnetic
        conductor (natural) or first-order Silver-Mueller absorbing boundary.
    hodge : {"voronoi", "barycentric"}
        Dual grid of the node masses: circumcentric (Voronoi, requires a
        Delaunay mesh with positive dual areas) or barycentric (A_T / 3 per
        vertex, positive on any mesh). The stiffness is the cotangent
        (Whitney) one in both cases.
    dt : float, optional
        Timestep [s]; default ``cfl`` times the leapfrog stability limit
        2 / sqrt(lambda_max(M^-1 K)).
    cfl : float, optional
        Fraction of the stability limit.

    Attributes
    ----------
    u : ndarray
        Ez (TM) [V/m] or Hz (TE) [A/m] on the mesh nodes.
    v : ndarray
        Magnetic (TM) [Wb/m] or electric (TE) [C/m] flux through primal edges.
    """

    def __init__(
        self,
        mesh,
        materials=None,
        polarization="TM",
        bc="pec",
        hodge="voronoi",
        dt=None,
        cfl=0.95,
    ):
        if mesh.dim != 2:
            raise ValueError("UnstructuredFIT2D needs a triangular mesh")
        if polarization not in ("TM", "TE"):
            raise ValueError("polarization must be 'TM' or 'TE'")
        if bc not in ("pec", "pmc", "abc"):
            raise ValueError("bc must be 'pec', 'pmc' or 'abc'")
        self.mesh, self.pol, self.bc, self.hodge = mesh, polarization, bc, hodge
        eps_r, mu_r, sigma, pec, dispersive = cell_materials(mesh, materials)
        self.eps_r, self.mu_r, self.sigma, self.pec_cells = eps_r, mu_r, sigma, pec
        self.dispersive = dispersive
        if polarization == "TE" and np.any(sigma > 0):
            raise NotImplementedError("conductivity in TE: use a Drude pole")

        if hodge == "voronoi":
            Pn = mesh.dual_area_partial
        elif hodge == "barycentric":
            Pn = mesh.dual_area_barycentric
        else:
            raise ValueError("hodge must be 'voronoi' or 'barycentric'")
        self._Pn = Pn
        L = mesh.edge_length
        He = mesh.dual_edge_partial  # (n_edges, n_cells)

        # Fixed nodes: Ez = 0 on PEC (TM); for TE, PEC is natural, but
        # nodes interior to PEC cells carry no field
        fixed = np.zeros(mesh.n_nodes, bool)
        used = np.zeros(mesh.n_nodes, bool)
        used[mesh.cells[~pec].ravel()] = True
        if polarization == "TM":
            fixed[mesh.cells[pec].ravel()] = True
            if bc == "pec":
                fixed[mesh.boundary_nodes] = True
        fixed |= ~used
        self.free = np.flatnonzero(~fixed)
        live = ~pec

        # Boundary dual length per node (half of each adjacent boundary edge)
        bl = np.zeros(mesh.n_nodes)
        be = mesh.boundary_edges
        np.add.at(bl, mesh.edges[be, 0], 0.5 * L[be])
        np.add.at(bl, mesh.edges[be, 1], 0.5 * L[be])
        # Wave impedance of the cell at each boundary edge
        cell_of_edge = np.full(mesh.n_edges, -1)
        cell_of_edge[mesh.cell_edges.ravel()] = np.repeat(np.arange(mesh.n_cells), 3)
        eta_node = np.full(mesh.n_nodes, ETA_0)
        ce = cell_of_edge[be]
        eta_b = ETA_0 * np.sqrt(mu_r[ce] / eps_r[ce])
        eta_node[mesh.edges[be, 0]] = eta_b
        eta_node[mesh.edges[be, 1]] = eta_b

        G = mesh.G[:, self.free]
        self.G = G
        if polarization == "TM":
            mass = eps_0 * (Pn @ (eps_r * live))
            stiff = (He @ (live / (mu_0 * mu_r))) / L
            loss = Pn @ (sigma * live)
            if bc == "abc":
                loss = loss + bl / eta_node
            groups = []
            for mat, sel in dispersive:
                c = eps_0 * (Pn @ (sel & live).astype(float))
                idx = np.flatnonzero(c[self.free] > 0)
                groups.append((mat, idx, c[self.free][idx]))
        else:
            mass = mu_0 * (Pn @ (mu_r * live))
            loss = bl * eta_node if bc == "abc" else np.zeros(mesh.n_nodes)
            # Series dual edges: per (edge, cell) partial lengths
            stiff = (He @ (live / (eps_0 * eps_r))) / L
            groups = []
        self.mass = mass[self.free]
        self.stiff = stiff
        self.loss = loss[self.free]
        if np.any(self.mass <= 0):
            raise ValueError(
                f"{np.sum(self.mass <= 0)} non-positive node masses: the mesh "
                "is not Delaunay near the boundary; use hodge='barycentric'"
            )
        self.K = (G.T @ sp.diags(stiff) @ G).tocsr()

        # Stability limit of the non-dispersive leapfrog
        lam = max_eigenvalue(self.K, M_inv_sqrt=1 / np.sqrt(self.mass))
        self.dt_max = 2 / np.sqrt(lam)
        self.dt = cfl * self.dt_max if dt is None else float(dt)
        h = mesh.edge_length.mean()
        self.courant = self.dt * np.sqrt(lam) / 2
        if dispersive:
            check_dispersive_stability(
                [m for m, _ in dispersive], self.dt, self.courant
            )
        self.h_mean = h

        if polarization == "TM":
            self._update = CapacitiveUpdate(self.mass, self.loss, groups, self.dt)
        else:
            self._init_te_dispersive(He, L, live)
        self.sources = []
        self.recorder = Recorder()
        self.reset()

    # ------------------------------------------------------------ TE series ADE
    def _init_te_dispersive(self, He, L, live):
        """Series ADE on the dual-edge segments inside dispersive triangles."""
        mesh = self.mesh
        self._te_groups = []
        if not self.dispersive:
            return
        # Non-dispersive part of the dual-edge voltage: v = W0 d
        disp = np.zeros(mesh.n_cells, bool)
        for _, sel in self.dispersive:
            disp |= sel
        self.stiff = (He @ ((live & ~disp) / (eps_0 * self.eps_r))) / L
        Hc = He.tocoo()
        for mat, sel in self.dispersive:
            keep = sel[Hc.col] & live[Hc.col]
            e, h = Hc.row[keep], Hc.data[keep]
            lorentz, debye = _ade_coefficients(mat, self.dt)
            beta = sum(b for _, b in debye)
            z = np.zeros(e.size)
            self._te_groups.append(
                {
                    "e": e,
                    "h": h,
                    "L": L[e],
                    "eps_inf": mat.eps_inf,
                    "beta": beta,
                    "E": z.copy(),
                    "lorentz": [(*cf, z.copy(), z.copy()) for cf in lorentz],
                    "debye": [(*cf, z.copy()) for cf in debye],
                }
            )

    def _te_voltage(self, d):
        """Dual-edge voltages v_e = sum_T h_{e,T} E_{e,T} [V/m * m]."""
        v = self.stiff * d
        for g in self._te_groups:
            v += np.bincount(g["e"], g["h"] * g["E"], minlength=v.size)
        return v

    def _te_ade_step(self, d):
        """Advance the segment fields E_{e,T} of dispersive triangles to n+1."""
        for g in self._te_groups:
            En = g["E"]
            Dn = d[g["e"]] / (g["L"] * eps_0)  # D / eps_0 normal to the edge
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
        """Zero all fields, auxiliary polarizations, probes and DFTs."""
        self.u = np.zeros(self.free.size)
        self.v = np.zeros(self.mesh.n_edges)
        self.n = 0
        if self.pol == "TM":
            self._update.reset()
        for g in getattr(self, "_te_groups", []):
            g["E"][:] = 0
            g["lorentz"] = [(a, x, z, 0 * p, 0 * q) for a, x, z, p, q in g["lorentz"]]
            g["debye"] = [(k, b, 0 * p) for k, b, p in g["debye"]]
        for rec in self.recorder.probes:
            rec["samples"] = []
        for rec in self.recorder.dfts:
            rec["data"][:] = 0

    @property
    def node_field(self):
        """Ez (TM) or Hz (TE) on all mesh nodes (0 on fixed nodes)."""
        out = np.zeros(self.mesh.n_nodes)
        out[self.free] = self.u
        return out

    def energy(self):
        """Discrete electromagnetic energy per unit length [J/m] (non-dispersive)."""
        if self.pol == "TM":
            return 0.5 * self.mass @ self.u**2 + 0.5 * self.stiff @ self.v**2
        return 0.5 * self.mass @ self.u**2 + 0.5 * self.stiff @ self.v**2

    # ------------------------------------------------------------ sources
    def _interp(self, points):
        """Sparse (n_points x n_free) linear interpolation of node values."""
        cell, lam = self.mesh.locate(points)
        if np.any(cell < 0):
            raise ValueError("point outside the mesh")
        rows = np.repeat(np.arange(len(cell)), 3)
        nodes = self.mesh.cells[cell].ravel()
        full = sp.csr_matrix(
            (lam.ravel(), (rows, nodes)), shape=(len(cell), self.mesh.n_nodes)
        )
        return full[:, self.free]

    def add_point_source(self, position, amp_func, direction=None):
        """
        Soft line current (per unit length in z) at a point.

        TM: electric line current I_z(t) [A] along z, distributed to the nodes
        with the barycentric (P1) weights of `position`. TE with `direction`:
        in-plane electric dipole moment per length p(t) (amp_func gives
        I l [A m / m = A]) projected on the edges with the Raviart-Thomas
        functions; TE without `direction`: magnetic line current K_z(t) [V]
        on the nodes.

        Parameters
        ----------
        position : (2,) float
            Source location [m].
        amp_func : callable
            t [s] -> amplitude.
        direction : (2,) float, optional
            Dipole direction (TE electric dipole only).
        """
        if direction is None:
            w = np.asarray(self._interp([position]).todense()).ravel()
            self.sources.append(("node", w, amp_func))
            return
        if self.pol != "TE":
            raise ValueError("an in-plane dipole needs polarization='TE'")
        d = np.asarray(direction, float) / np.linalg.norm(direction)
        cell, W = self.mesh.whitney_edge([position])
        # Raviart-Thomas (flux) function: W_e x z, oriented along t x z
        RT = np.stack([W[0, :, 1], -W[0, :, 0]], 1)
        w = np.zeros(self.mesh.n_edges)
        w[self.mesh.cell_edges[cell[0]]] = RT @ d
        self.sources.append(("edge", w, amp_func))

    # ------------------------------------------------------------ monitors
    def add_probe(self, points):
        """Record Ez (TM) / Hz (TE) at `points` every step; see `probe_signal`."""
        return self.recorder.add_probe(self._interp(points))

    def add_dft(self, freqs):
        """Running DFT of the node field at frequencies `freqs` [Hz]."""
        return self.recorder.add_dft(freqs, self.free.size)

    def probe_times(self):
        """Sample times of probes [s] (field at the end of each step)."""
        return (np.arange(self.n) + 1) * self.dt

    # ------------------------------------------------------------ time stepping
    def one_step(self):
        """Advance one leapfrog step (sources at the half step)."""
        dt = self.dt
        t_half = (self.n + 0.5) * dt
        node_src = np.zeros(self.free.size)
        edge_src = np.zeros(self.mesh.n_edges)
        for kind, w, f in self.sources:
            if kind == "node":
                node_src += w * f(t_half)
            else:
                edge_src += w * f(t_half)
        if self.pol == "TM":
            self.v -= dt * (self.G @ self.u)
            r = self.G.T @ (self.stiff * self.v) - node_src
            self.u = self._update.step(self.u, r)
        else:
            # hz at n+1/2 from v^n, then d to n+1 (leapfrog with roles swapped)
            vol = self._te_voltage(self.v)
            a = self.mass - self.loss * dt / 2
            b = self.mass + self.loss * dt / 2
            self.u = (a * self.u - dt * (self.G.T @ vol + node_src)) / b
            self.v += dt * (self.G @ self.u - edge_src)
            self._te_ade_step(self.v)
        self.n += 1
        self.recorder.record(self.u, self.n * dt, dt)

    def run(self, n_steps):
        """Advance `n_steps` steps."""
        for _ in range(int(n_steps)):
            self.one_step()

    def probe_signal(self, probe):
        """Recorded samples of a probe, shape (n_steps, n_points)."""
        return np.asarray(probe["samples"])

    # ------------------------------------------------------------ eigenmodes
    def eigenmodes(self, n_modes=6, f_target=0.0, frequency=None):
        """
        Resonant frequencies of the closed (PEC/PMC) structure.

        Solves K u = omega^2 M u by shift-invert Lanczos around `f_target`.
        Dispersive materials are evaluated at `frequency` (real permittivity,
        TE only) so that a fixed-point iteration can find dispersive modes,
        see `dispersive_eigenmode`.

        Parameters
        ----------
        n_modes : int
            Number of modes.
        f_target : float
            Shift frequency [Hz]; modes closest to it are returned.
        frequency : float, optional
            Frequency [Hz] at which dispersive permittivities are evaluated.

        Returns
        -------
        f : ndarray
            Mode frequencies [Hz] sorted ascending (TE: the constant mode at
            0 Hz is skipped).
        modes : ndarray (n_nodes, n_modes)
            Node fields.
        """
        K = self._stiffness_at(frequency)
        M = sp.diags(self.mass)
        sigma = (2 * np.pi * f_target) ** 2
        if sigma == 0:
            sigma = -1e-3 * (K.diagonal().mean() / self.mass.mean())
        k = n_modes + (1 if self.pol == "TE" else 0)
        w2, vec = eigsh(K, k=k, M=M, sigma=sigma, which="LM")
        o = np.argsort(w2)
        w2, vec = w2[o], vec[:, o]
        keep = np.abs(w2) > 1e-8 * np.abs(w2).max()
        w2, vec = w2[keep][:n_modes], vec[:, keep][:, :n_modes]
        modes = np.zeros((self.mesh.n_nodes, vec.shape[1]))
        modes[self.free] = vec
        return np.sqrt(np.abs(w2)) * np.sign(w2) / (2 * np.pi), modes

    def _stiffness_at(self, frequency):
        if not self.dispersive or frequency is None:
            if self.dispersive and self.pol == "TE":
                raise ValueError("pass `frequency` for dispersive TE eigenmodes")
            return self.K
        if self.pol != "TE":
            raise NotImplementedError("dispersive eigenmodes are implemented for TE")
        stiff = self.stiff.copy()
        for (mat, _), g in zip(self.dispersive, self._te_groups):
            eps = np.real(mat.epsilon(frequency))
            stiff += np.bincount(
                g["e"], g["h"] / (g["L"] * eps_0 * eps), minlength=stiff.size
            )
        return (self.G.T @ sp.diags(stiff) @ self.G).tocsr()

    def dispersive_eigenmode(self, f_guess, n_iter=50, rtol=1e-10):
        """
        Self-consistent mode of a dispersive TE structure.

        Solves g(f) = f_mode(K(f)) - f = 0 by the secant method, where
        f_mode(K(f)) is the eigenfrequency closest to f of the problem with
        the (real part of the) permittivity frozen at f.

        Parameters
        ----------
        f_guess : float
            Initial frequency [Hz].

        Returns
        -------
        f : float
            Self-consistent mode frequency [Hz].
        """

        def g(f):
            return self.eigenmodes(1, f_target=f, frequency=f)[0][0] - f

        f0, f1 = f_guess, f_guess * (1 + 1e-3)
        g0 = g(f0)
        for _ in range(n_iter):
            g1 = g(f1)
            if abs(g1) < rtol * f1 or g1 == g0:
                return f1
            f0, f1, g0 = f1, f1 - g1 * (f1 - f0) / (g1 - g0), g1
        raise RuntimeError("dispersive_eigenmode did not converge")
