# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Exact discrete adjoint of the unstructured FIT solvers.

`UnstructuredFIT2D` (TM, TE) and `UnstructuredFIT3D` (Voronoi and Whitney
Hodge) advance a state x^n = (node/edge field u, flux v or b, ADE
polarizations, PML filter memories) by a map that is affine in x and linear
in the per-cell permittivity eps_T only through fixed diagonal (or Whitney)
coefficients:

    x^{n+1} = A(eps) x^n + s^n.

For an objective F(X) of DFT samples X = sum_n c_n W u^{n+1}, the gradient is
the reverse sweep lam^n = A^T lam^{n+1} + (adjoint source), with one design
row per step. This module re-expresses each solver step as a pure function
on a state dictionary (bitwise equal to `one_step`), writes its exact
transpose term by term, and drives both with the memory rungs of
`wakis.inverse.memory` (stored rows or binomial checkpointing, bitwise equal).

The derivation of every transposed term, the design sensitivities and the
limits are in the physics guide section "Unstructured adjoint"
(``docs/physicsguide_sections/unstructured_adjoint.md``).

Classes
-------
UnstructuredAdjointStepper
    Forward step, its transpose and the inner product of one solver.
CellMaterialGrid
    Per-cell density -> permittivity with a cone or Helmholtz filter on the
    cell graph and a tanh projection.
UnstructuredAdjointSimulation
    Forward run and gradient; duck-type compatible with
    `wakis.inverse.jax_simulation` (JAX ``custom_vjp``).
CellDFTMonitor
    Running DFT of interpolated node (2D) or Whitney-edge (3D) fields.
"""

import numpy as np
import scipy.sparse as sp
from scipy.constants import epsilon_0 as eps_0
from scipy.sparse.linalg import splu

from ..inverse import memory as mem
from ..inverse.projection import tanh_projection, tanh_projection_derivative
from ._core import CapacitiveUpdate

__all__ = [
    "UnstructuredAdjointStepper",
    "UnstructuredAdjointSimulation",
    "CellMaterialGrid",
    "CellDFTMonitor",
    "whitney_local_mass",
]


# ============================================================ building blocks
class _Capacitive:
    """
    Functional form of `CapacitiveUpdate` (same coefficients and arithmetic).

    Forward (per dof k; groups g with dofs idx and vacuum capacitance c)::

        acc   = a1 u + dt r - sum_g c [sum_L (p'_L - p_L) + sum_D (kappa - 1) p_D]
        p'_L  = alpha p_L + xi p_L,old + zeta u[idx],   p'_L,old = p_L
        u'    = acc / den
        p'_D  = kappa p_D + beta (u'[idx] + u[idx])

    with a1 = C0 - G dt/2 - sum c beta, den = C0 + G dt/2 + sum c beta.
    """

    def __init__(self, C0, G, groups, dt):
        cu = CapacitiveUpdate(C0, G, groups, dt)
        self.dt, self.a1, self.den = dt, cu.a1, cu.den
        self.groups = [
            {
                "idx": g["idx"],
                "c": g["c"],
                "lorentz": [cf[:3] for cf in g["lorentz"]],
                "debye": [cf[:2] for cf in g["debye"]],
            }
            for g in cu.groups
        ]

    def keys(self, pre):
        out = []
        for gi, g in enumerate(self.groups):
            n = g["idx"].size
            out += [(f"{pre}{gi}L{k}{s}", n) for k in range(len(g["lorentz"])) for s in "po"]
            out += [(f"{pre}{gi}D{k}", n) for k in range(len(g["debye"]))]
        return out

    def forward(self, u, r, x, x1, pre):
        acc = self.a1 * u + self.dt * r
        for gi, g in enumerate(self.groups):
            idx, c = g["idx"], g["c"]
            un = u[idx]
            corr = np.zeros(idx.size)
            for k, (alpha, xi, zeta) in enumerate(g["lorentz"]):
                p, po = x[f"{pre}{gi}L{k}p"], x[f"{pre}{gi}L{k}o"]
                p_new = alpha * p + xi * po + zeta * un
                corr += p_new - p
                x1[f"{pre}{gi}L{k}p"], x1[f"{pre}{gi}L{k}o"] = p_new, p
            for k, (kappa, _) in enumerate(g["debye"]):
                corr += (kappa - 1) * x[f"{pre}{gi}D{k}"]
            np.add.at(acc, idx, -c * corr)
        u_new = acc / self.den
        for gi, g in enumerate(self.groups):
            idx = g["idx"]
            for k, (kappa, beta) in enumerate(g["debye"]):
                p = x[f"{pre}{gi}D{k}"]
                x1[f"{pre}{gi}D{k}"] = kappa * p + beta * (u_new[idx] + u[idx])
        return u_new

    def transpose(self, ub_new, xb1, xb, pre):
        """
        Adjoint of `forward`: from the adjoints of (u', states') return
        (adjoint of u, adjoint of r, total adjoint of u'), writing the
        adjoints of the input states into `xb`.
        """
        ub_new = np.array(ub_new, dtype=float, copy=True)
        ub = np.zeros_like(ub_new)
        for gi, g in enumerate(self.groups):
            idx = g["idx"]
            for k, (kappa, beta) in enumerate(g["debye"]):
                pb1 = xb1[f"{pre}{gi}D{k}"]
                xb[f"{pre}{gi}D{k}"] = kappa * pb1
                np.add.at(ub_new, idx, beta * pb1)
                np.add.at(ub, idx, beta * pb1)
        accb = ub_new / self.den
        ub += self.a1 * accb
        rb = self.dt * accb
        for gi, g in enumerate(self.groups):
            idx, c = g["idx"], g["c"]
            corrb = -c * accb[idx]
            unb = np.zeros(idx.size)
            for k, (kappa, _) in enumerate(g["debye"]):
                xb[f"{pre}{gi}D{k}"] = xb[f"{pre}{gi}D{k}"] + (kappa - 1) * corrb
            for k, (alpha, xi, zeta) in enumerate(g["lorentz"]):
                pnb = xb1[f"{pre}{gi}L{k}p"] + corrb
                xb[f"{pre}{gi}L{k}p"] = xb1[f"{pre}{gi}L{k}o"] - corrb + alpha * pnb
                xb[f"{pre}{gi}L{k}o"] = xi * pnb
                unb += zeta * pnb
            np.add.at(ub, idx, unb)
        return ub, rb, ub_new


class _CFS:
    """
    Functional form of the PML filter `solver2d._CFSRatio`::

        Px' = c1 Px + c2 (x + x0)
        y   = (kn x + sn Px' - sd (c1 Py + c2 y0)) / (kd + sd c2)
        Py' = c1 Py + c2 (y + y0),   x0' = x,   y0' = y
    """

    def __init__(self, f, name):
        self.f, self.name = f, name
        self.n = np.size(f.kn)

    def keys(self):
        return [(self.name + s, self.n) for s in ("Px", "Py", "x0", "y0")]

    def forward(self, x_in, x, x1):
        f, k = self.f, self.name
        c1, c2 = f.c1, f.c2
        Px = c1 * x[k + "Px"] + c2 * (x_in + x[k + "x0"])
        y = (f.kn * x_in + f.sn * Px - f.sd * (c1 * x[k + "Py"] + c2 * x[k + "y0"])) / (
            f.kd + f.sd * c2
        )
        x1[k + "Px"] = Px
        x1[k + "Py"] = c1 * x[k + "Py"] + c2 * (y + x[k + "y0"])
        x1[k + "x0"] = x_in.copy()
        x1[k + "y0"] = y
        return y

    def transpose(self, yb_ext, xb1, xb):
        f, k = self.f, self.name
        c1, c2 = f.c1, f.c2
        Pyb1 = xb1[k + "Py"]
        yb = yb_ext + xb1[k + "y0"] + c2 * Pyb1
        Pyb = c1 * Pyb1
        y0b = c2 * Pyb1
        t = yb / (f.kd + f.sd * c2)
        xinb = f.kn * t
        Pxb1 = xb1[k + "Px"] + f.sn * t
        Pyb = Pyb - f.sd * c1 * t
        y0b = y0b - f.sd * c2 * t
        xb[k + "Px"] = c1 * Pxb1
        xinb = xinb + c2 * Pxb1
        xb[k + "x0"] = c2 * Pxb1
        xb[k + "Py"] = Pyb
        xb[k + "y0"] = y0b
        return xinb + xb1[k + "x0"]


def whitney_local_mass(mesh):
    """
    Element Whitney 1-form mass matrices per unit relative permittivity.

    Returns ``M_loc`` (n_cells, 6, 6) with
    ``M1 = sum_T eps_T live_T M_loc[T]`` assembled on `mesh.cell_edges`
    (the integrand of `solver3d.whitney_mass_matrices`, including eps_0, the
    volume and the global edge orientation signs).
    """
    grad = mesh.barycentric_gradients()
    V = mesh.cell_volume
    Mll = (np.ones((4, 4)) + np.eye(4)) / 20
    loc_e = mesh._local_edges
    sgn = np.stack(
        [np.where(mesh.cells[:, a] < mesh.cells[:, b], 1.0, -1.0) for a, b in loc_e], 1
    )

    def dot(i, j):
        return np.einsum("ni,ni->n", grad[:, i], grad[:, j])

    out = np.zeros((mesh.n_cells, 6, 6))
    for p, (a, b) in enumerate(loc_e):
        for q, (c, d) in enumerate(loc_e):
            val = (
                Mll[a, c] * dot(b, d)
                - Mll[a, d] * dot(b, c)
                - Mll[b, c] * dot(a, d)
                + Mll[b, d] * dot(a, c)
            )
            out[:, p, q] = eps_0 * V * val * sgn[:, p] * sgn[:, q]
    return out


def _inner(x, y):
    return float(sum(np.vdot(x[k], y[k]).real for k in x))


# ============================================================ the stepper
class UnstructuredAdjointStepper:
    """
    Forward step, exact transpose and design sensitivities of one solver.

    The state is a dict of 1D arrays: ``u`` (free node field in 2D, free
    edge voltages in 3D), ``v`` (2D edge flux) or ``b`` (3D face flux), the
    ADE polarizations of the dispersive groups and, with the 2D radial PML,
    the node flux ``q`` and the memories of the four CFS filters.

    Parameters
    ----------
    solver : UnstructuredFIT2D or UnstructuredFIT3D
        A built solver (its operators, dt, LTS partition, PML and massless
        edges are frozen and reused; only the permittivity of the design
        cells changes through `set_eps`).
    design_cells : array_like of int, optional
        Cells whose permittivity is a design variable.

    Notes
    -----
    `step` reproduces `solver.one_step` bitwise (same operations in the same
    order; checked by the tests). `step_T` is its exact transpose with the
    sources removed; ``inner(step(x), y) == inner(x, step_T(y))`` to
    round-off for any x, y.
    """

    def __init__(self, solver, design_cells=None):
        self.solver = s = solver
        self.mesh = s.mesh
        self.dt = s.dt
        self.dim = s.mesh.dim
        if self.dim == 2:
            self.pol = s.pol
            self.kind = "cap" if s.pol == "TM" else "stiff"
            self.pml = s._pml
            if self.pml is not None:
                p = self.pml
                self._f = {
                    "fn1": _CFS(p.fn1, "fn1"),
                    "fn2": _CFS(p.fn2, "fn2"),
                    "fs": _CFS(p.fs, "fs"),
                    "fc": _CFS(p.fc, "fc"),
                }
            self.lts = s.lts
            self.absorber = s._absorber is not None
            if self.absorber:
                self.ce = 0.5 * s.dt * s.edge_rate
        else:
            self.pol = None
            self.kind = "cap" if s.hodge == "voronoi" else "whitney"
            self.pml = None
            self.lts = None
            self.massless = s.massless.size > 0
            # spherical CFS-PML (pml3d.PML3D): the step is S4 S3 S2 S1 on the
            # extended state [u, b, z]; S3 is rebuilt for every permittivity
            self.pml3 = getattr(s, "_pml", None)
            if self.pml3 is not None:
                L = self.pml3.layout
                self._nz = L.n - s.free.size - self.mesh.n_faces
        if self.dim == 2 and s.plane_wave is not None and s.dispersive:
            raise NotImplementedError(
                "dispersive plane wave (dispersive_planewave patches the solver ADEs)"
            )
        self.live = ~s.pec_cells
        self.base_eps = np.array(s.eps_r, dtype=float)
        self.design_cells = None
        if design_cells is not None:
            self._attach_design(np.asarray(design_cells, dtype=int))
        self.set_eps(self.base_eps)

    # ------------------------------------------------------------ coefficients
    def _groups(self):
        s = self.solver
        return [
            (mat, g["idx"], g["c"]) for (mat, _), g in zip(s.dispersive, s._update.groups)
        ]

    def set_eps(self, eps):
        """Rebuild the permittivity-dependent coefficients for per-cell `eps`."""
        s, mesh = self.solver, self.mesh
        eps = np.asarray(eps, dtype=float)
        self.eps = eps
        if self.dim == 2 and self.pol == "TM":
            mass = eps_0 * (s._Pn @ (eps * self.live))
            self.mass = mass[s.free]
            self.cu = _Capacitive(self.mass, s.loss, self._groups(), s.dt)
            self.stiff = s.stiff
        elif self.dim == 2:
            He, L = mesh.dual_edge_partial, mesh.edge_length
            if s.dispersive:
                disp = np.zeros(mesh.n_cells, bool)
                for _, sel in s.dispersive:
                    disp |= sel
                self.stiff = (He @ ((self.live & ~disp) / (eps_0 * eps))) / L
            else:
                self.stiff = (He @ (self.live / (eps_0 * eps))) / L
            self.mass = s.mass
        elif self.kind == "cap":
            live = self.live.astype(float)
            cap = eps_0 * mesh.dual_measure(eps * live) / mesh.edge_length
            self.cap = cap[s.free]
            if np.any(self.cap <= 0):
                raise ValueError("non-positive capacitance on a free edge")
            self.cu = _Capacitive(self.cap, s.cond, self._groups(), s.dt)
            if getattr(self, "pml3", None) is not None:
                p = self.pml3
                S1, S2, _, S4 = p.stages
                S3, B = p.ampere_stage(self.cu.a1, self.cu.den)
                self._pstages, self._pB = (S1, S2, S3, S4), B
                self._pstagesT = tuple(S.T.tocsr() for S in self._pstages)
        else:
            from .solver3d import whitney_mass_matrices

            M1, _ = whitney_mass_matrices(mesh, eps, s.mu_r, self.live.astype(float))
            self.M1 = M1[s.free][:, s.free].tocsc()
            self._M1 = splu(self.M1)
        if self.design_cells is not None:
            self._design_maps()

    # ------------------------------------------------------------ design
    def _attach_design(self, cells):
        s, mesh = self.solver, self.mesh
        self.design_cells = cells
        if np.any(s.pec_cells[cells]):
            raise ValueError("design cells must not be PEC")
        disp = np.zeros(mesh.n_cells, bool)
        for _, sel in s.dispersive:
            disp |= sel
        if np.any(disp[cells]):
            raise ValueError("design cells must not be dispersive")
        if getattr(s, "plane_wave", None) is not None:
            raise NotImplementedError(
                "plane-wave weights depend on the permittivity: no design gradient"
            )
        if self.dim == 2:
            bnodes = np.zeros(mesh.n_nodes, bool)
            bnodes[mesh.boundary_nodes] = True
            if np.any(bnodes[mesh.cells[cells]]):
                raise ValueError("design cells must not touch the outer boundary")
        else:
            bed = np.zeros(mesh.n_edges, bool)
            bed[mesh.boundary_edges] = True
            if np.any(bed[mesh.cell_edges[cells]]):
                raise ValueError("design cells must not touch the outer boundary")
            if getattr(self, "pml3", None) is not None:
                pe = np.zeros(mesh.n_edges, bool)
                pe[self.pml3.e_pml] = True
                if np.any(pe[mesh.cell_edges[cells]]):
                    raise ValueError("design cells inside the PML")
        pos = -np.ones(mesh.n_nodes if self.dim == 2 else mesh.n_edges, int)
        pos[s.free] = np.arange(s.free.size)
        if self.kind == "cap":
            ref = mesh.cells[cells] if self.dim == 2 else mesh.cell_edges[cells]
            loc = pos[np.unique(ref)]
            self.dofs = loc[loc >= 0]  # free dofs whose capacitance depends on eps
        elif self.kind == "stiff":
            self.dofs = np.unique(mesh.cell_edges[cells])  # design edges
        else:
            self.dofs = np.unique(pos[mesh.cell_edges[cells]])
            self.dofs = self.dofs[self.dofs >= 0]
            # local index of each design-cell edge in `dofs` (-1 -> zero slot)
            lookup = np.full(s.free.size + 1, self.dofs.size)
            lookup[self.dofs] = np.arange(self.dofs.size)
            p = pos[mesh.cell_edges[cells]]
            self._wloc = np.where(p >= 0, lookup[p], self.dofs.size)
            self._Mloc = whitney_local_mass(mesh)[cells]
        if self.dim == 2:
            if self.kind == "cap":
                nodes = s.free[self.dofs]
                if self.absorber and np.any(self._absorber_rate(mesh.nodes[nodes]) > 0):
                    raise ValueError("design nodes inside the absorber")
            if self.pml is not None:
                if self.kind == "cap" and np.isin(self.dofs, self.pml.nodes).any():
                    raise ValueError("design cells inside the PML")
                if self.kind == "stiff" and np.isin(
                    self.dofs, np.r_[self.pml.spokes, self.pml.chords]
                ).any():
                    raise ValueError("design cells inside the PML")
            if self.lts is not None:
                F = self.lts.F
                if self.kind == "cap" and np.isin(self.dofs, F).any():
                    raise ValueError("design nodes in the LTS fine region")
                if self.kind == "stiff":
                    inc = np.unique(s.G[:, F].tocsc().nonzero()[0])
                    if np.isin(self.dofs, inc).any():
                        raise ValueError("design edges touch the LTS fine region")

    def _absorber_rate(self, pts):
        a = self.solver._absorber
        xi = np.clip((np.hypot(*pts.T) - a["r0"]) / (a["r1"] - a["r0"]), 0, None)
        return a["s_max"] * xi ** a.get("order", 2)

    def _design_maps(self):
        """Sparse d(coefficient)/d(eps_T) on the design dofs (fixed per run)."""
        s, mesh, cells = self.solver, self.mesh, self.design_cells
        if self.kind == "cap":
            if self.dim == 2:
                P = s._Pn[s.free[self.dofs]][:, cells]
                self.dmap = (eps_0 * P).tocsr()
            else:
                P = mesh.dual_area_partial[s.free[self.dofs]][:, cells]
                Linv = sp.diags(1 / mesh.edge_length[s.free[self.dofs]])
                self.dmap = (eps_0 * Linv @ P).tocsr()
            self.dmap = self.dmap.multiply(self.live[cells][None, :]).tocsr()
        elif self.kind == "stiff":
            He = mesh.dual_edge_partial[self.dofs][:, cells]
            Linv = sp.diags(1 / mesh.edge_length[self.dofs])
            w = -1 / (eps_0 * self.eps[cells] ** 2)
            self.dmap = (Linv @ He @ sp.diags(w)).tocsr()

    def design_gradient_init(self):
        n = self.design_cells.size if self.kind == "whitney" else self.dofs.size
        return np.zeros(n)

    def accumulate(self, g, sens, row):
        """g += d F / d(coefficient) of one step from its sens and row."""
        if self.kind == "whitney":
            sz = np.r_[sens, 0.0][self._wloc]
            rz = np.r_[row, 0.0][self._wloc]
            g -= np.einsum("tp,tpq,tq->t", sz, self._Mloc, rz)
        else:
            g += sens * row

    def cell_gradient(self, g):
        """dF/deps_T on the design cells from the accumulated g."""
        if self.kind == "whitney":
            return g * self.live[self.design_cells]
        return self.dmap.T @ g

    # ------------------------------------------------------------ state
    def keys(self):
        s = self.solver
        if self.dim == 2:
            out = [("u", s.free.size), ("v", self.mesh.n_edges)]
            if self.pol == "TM":
                out += self.cu.keys("cu")
            else:
                for gi, g in enumerate(s._te_groups):
                    n = g["e"].size
                    out += [(f"te{gi}E", n)]
                    out += [(f"te{gi}L{k}{c}", n) for k in range(len(g["lorentz"])) for c in "po"]
                    out += [(f"te{gi}D{k}", n) for k in range(len(g["debye"]))]
            if self.pml is not None:
                out += [("q", self.pml.nodes.size)]
                for f in self._f.values():
                    out += f.keys()
        else:
            out = [("u", s.free.size), ("b", self.mesh.n_faces)]
            if self.kind == "cap":
                out += self.cu.keys("cu")
            if self.pml3 is not None:
                out += [("z", self._nz)]  # PML flux d and filter memories
        return out

    def zero_state(self):
        return {k: np.zeros(n) for k, n in self.keys()}

    def random_state(self, rng):
        return {k: rng.standard_normal(n) for k, n in self.keys()}

    inner = staticmethod(_inner)

    # ------------------------------------------------------------ sources
    def _sources(self, n):
        s, dt = self.solver, self.dt
        t_half = (n + 0.5) * dt
        if self.dim == 3:
            src = np.zeros(s.free.size)
            for w, f in s.sources:
                src += w * f(t_half)
            return src, None
        node_src = np.zeros(s.free.size)
        edge_src = np.zeros(self.mesh.n_edges)
        for kind, w, f in s.sources:
            if kind == "node":
                node_src += w * f(t_half)
            else:
                edge_src += w * f(t_half)
        if s.plane_wave is not None and s.pol == "TM":
            node_src += s._pw_values(n + 1) - s._pw_values(n)
        return node_src, edge_src

    # ------------------------------------------------------------ 2D helpers
    def _grad(self, u):
        return self.solver._grad(u)

    def _grad_T(self, z):
        y = self.solver.G.T @ z
        return y if self.lts is None else self.lts.apply_T(y)

    def _te_voltage(self, v, x):
        out = self.stiff * v
        for gi, g in enumerate(self.solver._te_groups):
            out += np.bincount(g["e"], g["h"] * x[f"te{gi}E"], minlength=out.size)
        return out

    def _te_ade(self, v1, x, x1):
        for gi, g in enumerate(self.solver._te_groups):
            En = x[f"te{gi}E"]
            Dn = v1[g["e"]] / (g["L"] * eps_0)
            psum = np.zeros(En.size)
            for k, (alpha, xi, zeta, _, _) in enumerate(g["lorentz"]):
                p, po = x[f"te{gi}L{k}p"], x[f"te{gi}L{k}o"]
                p_new = alpha * p + xi * po + zeta * En
                psum += p_new
                x1[f"te{gi}L{k}p"], x1[f"te{gi}L{k}o"] = p_new, p
            for k, (kappa, beta, _) in enumerate(g["debye"]):
                psum += kappa * x[f"te{gi}D{k}"] + beta * En
            E_new = (Dn - psum) / (g["eps_inf"] + g["beta"])
            for k, (kappa, beta, _) in enumerate(g["debye"]):
                x1[f"te{gi}D{k}"] = kappa * x[f"te{gi}D{k}"] + beta * (E_new + En)
            x1[f"te{gi}E"] = E_new

    def _te_ade_T(self, xb1, xb, v1b):
        """Adjoint of `_te_ade`: adds to v1b, writes the ADE-path adjoints of E^n."""
        for gi, g in enumerate(self.solver._te_groups):
            Eb_new = np.array(xb1[f"te{gi}E"], copy=True)
            Enb = np.zeros(Eb_new.size)
            for k, (kappa, beta, _) in enumerate(g["debye"]):
                pb1 = xb1[f"te{gi}D{k}"]
                xb[f"te{gi}D{k}"] = kappa * pb1
                Eb_new += beta * pb1
                Enb += beta * pb1
            den = g["eps_inf"] + g["beta"]
            Dnb = Eb_new / den
            psumb = -Eb_new / den
            for k, (kappa, beta, _) in enumerate(g["debye"]):
                xb[f"te{gi}D{k}"] = xb[f"te{gi}D{k}"] + kappa * psumb
                Enb += beta * psumb
            for k, (alpha, xi, zeta, _, _) in enumerate(g["lorentz"]):
                pnb = xb1[f"te{gi}L{k}p"] + psumb
                xb[f"te{gi}L{k}p"] = xb1[f"te{gi}L{k}o"] + alpha * pnb
                xb[f"te{gi}L{k}o"] = xi * pnb
                Enb += zeta * pnb
            v1b += np.bincount(g["e"], Dnb / (g["L"] * eps_0), minlength=v1b.size)
            xb[f"te{gi}E"] = Enb

    def _pml_nodes(self, q1, x, x1):
        mass = self.mass[self.pml.nodes]
        return self._f["fn2"].forward(self._f["fn1"].forward(q1 / mass, x, x1), x, x1)

    def _pml_nodes_T(self, yb, xb1, xb):
        mass = self.mass[self.pml.nodes]
        return self._f["fn1"].transpose(self._f["fn2"].transpose(yb, xb1, xb), xb1, xb) / mass

    # ------------------------------------------------------------ forward step
    def step(self, x, n, sources=True, want_row=False):
        """
        One step x^n -> x^{n+1} (`one_step` of the solver, bitwise).

        Returns
        -------
        x1 : dict
            New state.
        row : ndarray or None
            The design row of this step (`want_row` and a design).
        """
        if sources:
            src, edge_src = self._sources(n)
        else:
            src = np.zeros(self.solver.free.size)
            edge_src = None if self.dim == 3 else np.zeros(self.mesh.n_edges)
        x1 = {}
        row = None
        if self.dim == 3:
            u1 = self._step3d(x, x1, src)
            if want_row and self.design_cells is not None:
                if self.kind == "cap":
                    row = ((x["u"] - u1) / self.cu.den)[self.dofs]
                else:
                    row = (u1 - x["u"])[self.dofs]
            return x1, row
        if self.pol == "TM":
            u1 = self._step_tm(x, x1, src)
            if want_row and self.design_cells is not None:
                row = ((x["u"] - u1) / self.cu.den)[self.dofs]
        else:
            if want_row and self.design_cells is not None:
                row = x["v"][self.dofs].copy()
            self._step_te(x, x1, src, edge_src, n, sources)
        return x1, row

    def _step_tm(self, x, x1, src):
        s, dt = self.solver, self.dt
        u, v = x["u"], x["v"]
        if self.pml is not None:
            p = self.pml
            sp_, ch = p.spokes, p.chords
            v1 = v - dt * self._grad(u)
            h = self.stiff * v1
            h[sp_] = self.stiff[sp_] * self._f["fs"].forward(v1[sp_], x, x1)
            h[ch] = self.stiff[ch] * self._f["fc"].forward(v1[ch], x, x1)
            r = s.G.T @ h - src
            u1 = self.cu.forward(u, r, x, x1, "cu")
            q1 = x["q"] + dt * r[p.nodes]
            u1[p.nodes] = self._pml_nodes(q1, x, x1)
            x1["q"] = q1
        else:
            if not self.absorber:
                v1 = v - dt * self._grad(u)
            else:
                ce = self.ce
                v1 = ((1 - ce) * v - dt * self._grad(u)) / (1 + ce)
            r = s.G.T @ (self.stiff * v1) - src
            u1 = self.cu.forward(u, r, x, x1, "cu")
        x1["u"], x1["v"] = u1, v1
        return u1

    def _step_te(self, x, x1, src, edge_src, n, sources):
        s, dt = self.solver, self.dt
        u, v = x["u"], x["v"]
        vol = self._te_voltage(v, x)
        if self.pml is not None:
            p = self.pml
            sp_, ch = p.spokes, p.chords
            vol[sp_] = self.stiff[sp_] * self._f["fs"].forward(v[sp_], x, x1)
            vol[ch] = self.stiff[ch] * self._f["fc"].forward(v[ch], x, x1)
        if s.plane_wave is not None and sources:
            vol -= s._pw_values(n)
        a = s.mass - s.loss * dt / 2
        b = s.mass + s.loss * dt / 2
        if self.pml is not None:
            r = -(s.G.T @ vol + src)
            u1 = (a * u + dt * r) / b
            q1 = x["q"] + dt * r[p.nodes]
            u1[p.nodes] = self._pml_nodes(q1, x, x1)
            x1["q"] = q1
            v1 = v + dt * (self._grad(u1) - edge_src)
        else:
            u1 = (a * u - dt * (s.G.T @ vol + src)) / b
            if not self.absorber:
                v1 = v + dt * (self._grad(u1) - edge_src)
            else:
                ce = self.ce
                v1 = ((1 - ce) * v + dt * (self._grad(u1) - edge_src)) / (1 + ce)
        self._te_ade(v1, x, x1)
        x1["u"], x1["v"] = u1, v1

    def _massless(self, u):
        return -self.solver._Kss.solve(self.solver._Ksf @ u)

    def _massless_T(self, esb):
        """Transpose of u -> -K_ss^-1 K_sf u: -K_sf^T K_ss^-T esb."""
        s = self.solver
        return -(s._Ksf.T @ s._Kss.solve(esb, trans="T"))

    def _step3d_pml(self, x, x1, src):
        """`PML3D.apply` on [u, b, z] with this permittivity's S3 (bitwise)."""
        S1, S2, S3, S4 = self._pstages
        xe = np.concatenate([x["u"], x["b"], x["z"]])
        y = S3 @ (S2 @ (S1 @ xe)) + self._pB @ src
        y = S4 @ y
        nu, nf = x["u"].size, x["b"].size
        x1["u"], x1["b"], x1["z"] = y[:nu], y[nu : nu + nf], y[nu + nf :]
        return x1["u"]

    def _step3d_pml_T(self, xb1, xb):
        """(S4 S3 S2 S1)^T; returns the adjoint of u after S3 (design sens)."""
        T1, T2, T3, T4 = self._pstagesT
        le = np.concatenate([xb1["u"], xb1["b"], xb1["z"]])
        y = T4 @ le
        nu, nf = xb1["u"].size, xb1["b"].size
        sens_full = y[:nu].copy()
        z = T1 @ (T2 @ (T3 @ y))
        xb["u"], xb["b"], xb["z"] = z[:nu], z[nu : nu + nf], z[nu + nf :]
        return sens_full

    def _step3d(self, x, x1, src):
        if self.pml3 is not None:
            return self._step3d_pml(x, x1, src)
        s, dt = self.solver, self.dt
        u, b = x["u"], x["b"]
        e_s = self._massless(u) if self.massless else None
        b1 = b - dt * (s.Cf @ u)
        if e_s is not None:
            b1 = b1 - dt * (s.Cs @ e_s)
        r = s.Cf.T @ (s.Mnu @ b1) - src
        if self.kind == "cap":
            u1 = self.cu.forward(u, r, x, x1, "cu")
        else:
            u1 = u + dt * self._M1.solve(r)
        x1["u"], x1["b"] = u1, b1
        return u1

    # ------------------------------------------------------------ transpose
    def step_T(self, xb1):
        """
        Exact transpose of the source-free `step`.

        Returns
        -------
        xb : dict
            A^T xb1.
        sens : ndarray or None
            Per-dof sensitivity of this step, combined with the forward row by
            `accumulate` (None without a design).
        """
        xb = {}
        if self.dim == 3:
            sens = self._step3d_T(xb1, xb)
        elif self.pol == "TM":
            sens = self._step_tm_T(xb1, xb)
        else:
            sens = self._step_te_T(xb1, xb)
        if self.design_cells is None:
            sens = None
        return xb, sens

    def _step_tm_T(self, xb1, xb):
        s, dt = self.solver, self.dt
        u1b, v1b = xb1["u"], xb1["v"]
        if self.pml is not None:
            p = self.pml
            sp_, ch = p.spokes, p.chords
            utb = np.array(u1b, copy=True)
            utb[p.nodes] = 0.0
            q1b = xb1["q"] + self._pml_nodes_T(u1b[p.nodes], xb1, xb)
            xb["q"] = q1b
            ub, rb, sens_full = self.cu.transpose(utb, xb1, xb, "cu")
            rb[p.nodes] += dt * q1b
            hb = s.G @ rb
            sb = self.stiff * hb
            v1b_tot = v1b + sb
            v1b_tot[sp_] = v1b[sp_] + self._f["fs"].transpose(sb[sp_], xb1, xb)
            v1b_tot[ch] = v1b[ch] + self._f["fc"].transpose(sb[ch], xb1, xb)
            xb["v"] = v1b_tot
            ub = ub - self._grad_T(dt * v1b_tot)
        else:
            ub, rb, sens_full = self.cu.transpose(u1b, xb1, xb, "cu")
            v1b_tot = v1b + self.stiff * (s.G @ rb)
            if not self.absorber:
                xb["v"] = v1b_tot
                ub = ub - self._grad_T(dt * v1b_tot)
            else:
                ce = self.ce
                xb["v"] = (1 - ce) / (1 + ce) * v1b_tot
                ub = ub - self._grad_T(dt * v1b_tot / (1 + ce))
        xb["u"] = ub
        return None if self.design_cells is None else sens_full[self.dofs]

    def _step_te_T(self, xb1, xb):
        s, dt = self.solver, self.dt
        v1b = np.array(xb1["v"], copy=True)
        self._te_ade_T(xb1, xb, v1b)
        a = s.mass - s.loss * dt / 2
        b = s.mass + s.loss * dt / 2
        if self.pml is not None:
            p = self.pml
            sp_, ch = p.spokes, p.chords
            vb = v1b
            u1b = xb1["u"] + self._grad_T(dt * v1b)
            utb = np.array(u1b, copy=True)
            utb[p.nodes] = 0.0
            q1b = xb1["q"] + self._pml_nodes_T(u1b[p.nodes], xb1, xb)
            xb["q"] = q1b
            ub = a / b * utb
            rb = dt * utb / b
            rb[p.nodes] += dt * q1b
            volb = -(s.G @ rb)
            mask = np.ones(volb.size)
            mask[sp_] = 0.0
            mask[ch] = 0.0
            volb_m = volb * mask
            vb = vb + self.stiff * volb_m
            vb[sp_] += self._f["fs"].transpose(self.stiff[sp_] * volb[sp_], xb1, xb)
            vb[ch] += self._f["fc"].transpose(self.stiff[ch] * volb[ch], xb1, xb)
        else:
            if not self.absorber:
                vb = v1b
                u1b = xb1["u"] + self._grad_T(dt * v1b)
            else:
                ce = self.ce
                vb = (1 - ce) / (1 + ce) * v1b
                u1b = xb1["u"] + self._grad_T(dt * v1b / (1 + ce))
            ub = a / b * u1b
            volb = -dt * (s.G @ (u1b / b))
            volb_m = volb
            vb = vb + self.stiff * volb
        for gi, g in enumerate(s._te_groups):
            xb[f"te{gi}E"] = xb[f"te{gi}E"] + g["h"] * volb_m[g["e"]]
        xb["u"], xb["v"] = ub, vb
        return None if self.design_cells is None else volb[self.dofs]

    def _step3d_T(self, xb1, xb):
        if self.pml3 is not None:
            sens_full = self._step3d_pml_T(xb1, xb)
            return None if self.design_cells is None else sens_full[self.dofs]
        s, dt = self.solver, self.dt
        u1b, b1b = xb1["u"], xb1["b"]
        if self.kind == "cap":
            ub, rb, sens_full = self.cu.transpose(u1b, xb1, xb, "cu")
        else:
            w = self._M1.solve(u1b, trans="T")
            ub = np.array(u1b, copy=True)
            rb = dt * w
            sens_full = w
        b1b_tot = b1b + s.Mnu.T @ (s.Cf @ rb)
        xb["b"] = b1b_tot
        ub = ub - dt * (s.Cf.T @ b1b_tot)
        if self.massless:
            ub = ub + self._massless_T(-dt * (s.Cs.T @ b1b_tot))
        xb["u"] = ub
        return None if self.design_cells is None else sens_full[self.dofs]


# ============================================================ design
class CellMaterialGrid:
    """
    Per-cell relative permittivity eps_T = eps_min + (eps_max - eps_min) P(F rho).

    Parameters
    ----------
    mesh : SimplexMesh
        The solver mesh.
    cells : array_like of int or bool
        Design cells (index array or mask over `mesh.n_cells`).
    eps_min, eps_max : float
        Relative permittivity at rho = 0 and rho = 1.
    filter_radius : float, optional
        Filter length R [m]; no filter if None.
    filter : {"cone", "helmholtz"}
        "cone": rho_f,i = sum_j w_ij A_j rho_j / sum_j w_ij A_j with the
        linear hat w_ij = max(0, 1 - |c_i - c_j| / R) on cell centroids c and
        areas (volumes) A. "helmholtz": the PDE filter
        -r^2 lap(rho_f) + rho_f = rho, r = R / (2 sqrt 3) (Lazarov & Sigmund
        2011), discretised by two-point fluxes on the cell adjacency graph
        (weight |shared facet| / |c_i - c_j|; zero flux out of the design
        region).
    beta : float, optional
        tanh projection steepness (none if None); `eta` its threshold.

    Notes
    -----
    Every stage is linear or pointwise, so `vjp` is exact: F^T (P' * dF/deps
    * (eps_max - eps_min)).
    """

    def __init__(self, mesh, cells, eps_min=1.0, eps_max=4.0, filter_radius=None,
                 filter="cone", beta=None, eta=0.5):  # fmt: skip
        cells = np.asarray(cells)
        if cells.dtype == bool:
            cells = np.flatnonzero(cells)
        self.mesh = mesh
        self.cells = cells.astype(int)
        self.eps_min, self.eps_max = float(eps_min), float(eps_max)
        self.beta, self.eta = beta, eta
        self.shape = (self.cells.size,)
        self.centroid = mesh.nodes[mesh.cells[self.cells]].mean(1)
        self.measure = np.abs(mesh.cell_volume[self.cells])
        self.F = None
        self._helm = None
        if filter_radius is not None:
            if filter == "cone":
                self.F = self._cone(float(filter_radius))
            elif filter == "helmholtz":
                self._helm = self._helmholtz(float(filter_radius))
            else:
                raise ValueError("filter must be 'cone' or 'helmholtz'")

    def _cone(self, R):
        from scipy.spatial import cKDTree

        n = self.cells.size
        pairs = cKDTree(self.centroid).query_pairs(R, output_type="ndarray")
        i, j = pairs[:, 0], pairs[:, 1]
        d = np.linalg.norm(self.centroid[i] - self.centroid[j], axis=1)
        w = 1 - d / R
        rows = np.r_[i, j, np.arange(n)]
        cols = np.r_[j, i, np.arange(n)]
        hat = np.r_[w, w, np.ones(n)]
        W = sp.csr_matrix((hat * self.measure[cols], (rows, cols)), shape=(n, n))
        return (sp.diags(1 / np.asarray(W.sum(1)).ravel()) @ W).tocsr()

    def _helmholtz(self, R):
        mesh = self.mesh
        r = R / (2 * np.sqrt(3))
        n = self.cells.size
        local = -np.ones(mesh.n_cells, int)
        local[self.cells] = np.arange(n)
        if mesh.dim == 2:
            facets, size = mesh.cell_edges[self.cells], mesh.edge_length
        else:
            facets, size = mesh.cell_faces[self.cells], mesh.face_area
        f = facets.ravel()
        owner = np.repeat(np.arange(n), facets.shape[1])
        o = np.argsort(f, kind="stable")
        f, owner = f[o], owner[o]
        pair = np.flatnonzero(f[1:] == f[:-1])
        i, j = owner[pair], owner[pair + 1]
        dist = np.linalg.norm(self.centroid[i] - self.centroid[j], axis=1)
        w = r**2 * size[f[pair]] / dist
        Lap = sp.csr_matrix((np.r_[-w, -w], (np.r_[i, j], np.r_[j, i])), shape=(n, n))
        Lap = Lap - sp.diags(np.asarray(Lap.sum(1)).ravel())
        S = (sp.diags(self.measure) + Lap).tocsc()
        return splu(S)

    def filtered(self, rho):
        rho = np.asarray(rho, dtype=float)
        if self.F is not None:
            return self.F @ rho
        if self._helm is not None:
            return self._helm.solve(self.measure * rho)
        return rho

    def _filter_T(self, g):
        if self.F is not None:
            return self.F.T @ g
        if self._helm is not None:
            return self.measure * self._helm.solve(g, trans="T")
        return g

    def projected(self, rho):
        rf = self.filtered(rho)
        return rf if self.beta is None else tanh_projection(rf, self.beta, self.eta)

    def eps_r(self, rho):
        """Relative permittivity of the design cells."""
        return self.eps_min + (self.eps_max - self.eps_min) * self.projected(rho)

    def cell_eps(self, rho, base_eps):
        """Full per-cell eps array with the design cells set from rho."""
        eps = np.array(base_eps, dtype=float, copy=True)
        eps[self.cells] = self.eps_r(rho)
        return eps

    def vjp(self, rho, grad_eps):
        """dF/drho from dF/deps_r on the design cells."""
        g = (self.eps_max - self.eps_min) * np.asarray(grad_eps, dtype=float)
        if self.beta is not None:
            g = g * tanh_projection_derivative(self.filtered(rho), self.beta, self.eta)
        return self._filter_T(g)


# ============================================================ monitors
class CellDFTMonitor:
    """
    Running DFT X[f, j] = sum_n exp(-2 pi i f t_{n+1}) dt (W u^{n+1})_j.

    The kernel, the dt factor and the sample times t_{n+1} = (n + 1) dt are
    those of `Recorder.add_dft` (field at the end of each step).

    Parameters
    ----------
    freqs : array_like
        Frequencies [Hz].
    points : (n, d) array_like, optional
        Sample points [m]: linear (2D) or Whitney (3D, `component`)
        interpolation of the solver field. Without points, all free dofs.
    component : {0, 1, 2}, optional
        3D only: Cartesian E component.
    """

    def __init__(self, freqs, points=None, component=None):
        self.freqs = np.atleast_1d(np.asarray(freqs, dtype=float))
        self.points = None if points is None else np.atleast_2d(points)
        self.component = component

    def weights(self, solver):
        if self.points is None:
            return sp.identity(solver.free.size, format="csr")
        if solver.mesh.dim == 2:
            return solver._interp(self.points).tocsr()
        if self.component is None:
            raise ValueError("3D point monitors need a component")
        return solver._whitney_interp(self.points, self.component).tocsr()


# ============================================================ simulation
class _StateIO:
    """capture/restore of the stepper state for `wakis.inverse.memory`."""

    def __init__(self, sim):
        self.sim = sim

    def capture(self):
        return {k: v.copy() for k, v in self.sim._x.items()}

    def restore(self, snap):
        self.sim._x = {k: v.copy() for k, v in snap.items()}


class UnstructuredAdjointSimulation:
    """
    Forward run and exact discrete-adjoint gradient on an unstructured mesh.

    Parameters
    ----------
    make_solver : callable
        Returns a fresh `UnstructuredFIT2D`/`UnstructuredFIT3D` with its
        sources. Its dt, LTS partition, PML and massless edges are frozen for
        every design: build it with the design cells at the *lowest*
        permittivity (eps_min) so that dt is stable for every rho (raising
        eps only lowers lambda_max of the leapfrog operator; see the guide).
    design : CellMaterialGrid
    monitors : dict of str to CellDFTMonitor
    Nt : int
        Number of steps.
    memory : {"store", "checkpoint"}
        "store": keep all Nt design rows. "checkpoint": binomial
        checkpointing of the full state (`wakis.inverse.memory.CheckpointRung`),
        bitwise-equal gradient.
    checkpoints, block : int, optional
        Snapshot slots and steps per block for "checkpoint" (default 4 slots,
        binom(slots + 2, 2) blocks).

    Notes
    -----
    `forward(rho)` returns ``{name: {"dft": (n_freq, n_points) complex}}``;
    `vjp(G)` returns dF/drho for G = dF/dRe X + 1j dF/dIm X. The object can
    be wrapped by `wakis.inverse.jax_simulation`.
    """

    def __init__(self, make_solver, design, monitors, Nt, memory="store",
                 checkpoints=None, block=None):  # fmt: skip
        if memory not in ("store", "checkpoint"):
            raise ValueError("memory must be 'store' or 'checkpoint'")
        self.make_solver = make_solver
        self.design = design
        self.monitors = dict(monitors)
        self.Nt = int(Nt)
        self.memory = memory
        self.checkpoints = checkpoints
        self.block = block
        self.verify_replay = False
        self.solver = make_solver()
        self.stepper = UnstructuredAdjointStepper(self.solver, design.cells)
        self._W = {k: m.weights(self.solver) for k, m in self.monitors.items()}
        self.rho = None
        self._recorded = False
        self.memory_stats = None

    def output_shapes(self):
        return {
            k: {"dft": (m.freqs.size, self._W[k].shape[0])}
            for k, m in self.monitors.items()
        }

    def _phase(self, freqs, n):
        return np.exp(-2j * np.pi * freqs * ((n + 1) * self.stepper.dt)) * self.stepper.dt

    def _make_rung(self, advance):
        st = self.stepper
        row_shape = (st.dofs.size,)
        if self.memory == "store":
            return mem.StoreRung(self.Nt, row_shape, np.float64, np, on_host=True)
        slots = self.checkpoints or 4
        block = self.block
        if block is None:
            block = max(1, -(-(self.Nt - 1) // mem.beta(slots, 2)))
        return mem.CheckpointRung(self.Nt, slots, block, advance, _StateIO(self),
                                  np.float64, np, verify=self.verify_replay)  # fmt: skip

    def forward(self, rho, record=True):
        """Run the simulation for density `rho`; returns the DFT outputs."""
        st = self.stepper
        rho = np.asarray(rho, dtype=float)
        st.set_eps(self.design.cell_eps(rho, st.base_eps))
        out = {k: np.zeros((m.freqs.size, self._W[k].shape[0]), complex)
               for k, m in self.monitors.items()}  # fmt: skip
        self._x = st.zero_state()

        def advance(n, want_row):
            self._x, row = st.step(self._x, n, want_row=want_row)
            return row

        rung = self._make_rung(advance) if record else mem.NullRung()
        for n in range(self.Nt):
            rung.before_step(n)
            self._x, row = st.step(self._x, n, want_row=rung.wants_row(n))
            u = self._x["u"]
            for k, m in self.monitors.items():
                out[k] += np.outer(self._phase(m.freqs, n), self._W[k] @ u)
            rung.after_step(n, row)
        rung.finish()
        self._rung = rung
        self.rho = rho.copy()
        self._recorded = bool(record)
        self.memory_stats = rung.stats()
        return {k: {"dft": v} for k, v in out.items()}

    def vjp(self, cotangents):
        """dF/drho by the reverse sweep, after `forward`."""
        if self.rho is None or not self._recorded:
            raise RuntimeError("Call forward(rho) (with record=True) before vjp.")
        st = self.stepper
        taps = []
        for k, comps in cotangents.items():
            G = np.asarray(comps["dft"], dtype=complex)
            taps.append((self.monitors[k].freqs, self._W[k].T.tocsr(), G))
        lam = st.zero_state()
        g = st.design_gradient_init()
        get_row = self._rung.reader()
        for n in reversed(range(self.Nt)):
            for freqs, WT, G in taps:
                w = np.conj(self._phase(freqs, n))
                lam["u"] = lam["u"] + WT @ np.real(w @ G)
            lam, sens = st.step_T(lam)
            st.accumulate(g, sens, get_row(n))
        self.memory_stats = self._rung.stats()
        return self.design.vjp(self.rho, st.cell_gradient(g))

    def value_and_grad(self, rho, objective):
        """
        F(X(rho)) and dF/drho for ``objective(outputs) -> (F, G)`` with G the
        cotangents (same structure as the outputs).
        """
        out = self.forward(rho)
        F, G = objective(out)
        return F, self.vjp(G)
