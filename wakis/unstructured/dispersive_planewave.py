# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Scattered-field plane wave on dispersive scatterers (`UnstructuredFIT2D`).

Derivation (per unit length in z; see also `UnstructuredFIT2D.add_plane_wave`).
The incident wave (E_i, H_i) solves Maxwell's equations in the lossless host
eps_b, so d_t (eps_0 eps_b E_i) = curl H_i. In a dispersive cell the total
fields obey d_t D = curl H with D = eps_0 eps_inf E + P and P = sum_p P_p
driven by the *total* E through the ADEs of `wakis.dispersive`. With
E = E_s + E_i, H = H_s + H_i and the scattered displacement
D_s = D - eps_0 eps_b E_i,

    d_t D_s = curl H_s,      d_t B_s = -curl E_s,
    D_s = eps_0 eps_inf E_s + eps_0 (eps_inf - eps_b) E_i + P[E_s + E_i].

Hence the instantaneous part is the non-dispersive scattered-field source
with eps -> eps_inf, and the poles are driven by the total field. Writing
the polarization current as dP/dt makes the equivalent source explicit:
J_eq = eps_0 (eps_inf - eps_b) dE_i/dt + dP[E_s + E_i]/dt, i.e. the
incident field polarises the metal, and that polarization current radiates
the scattered field.

* TM (E_z on nodes, tangential to every interface). The node update of
  `CapacitiveUpdate`, C0 (u' - u) + c sum_p (p' - p) = dt r, holds for the
  total field with C0 = eps_0 sum_T eps_inf,T A_{n,T}. Substituting
  u = u_s + E_i gives the existing source
  i_n = (C0 - eps_0 eps_b A_n) (E_i^{n+1} - E_i^n) / dt plus the ADE drive
  with the total node field: p^{n+1} = alpha p^n + xi p^{n-1} +
  zeta (u_s^n + E_i^n) (Lorentz/Drude) and beta (u'^{n+1} + u^n) with
  u = u_s + E_i (Debye).
* TE (in-plane E, crosses interfaces). Each dual-edge segment (e, T) of a
  dispersive triangle carries the total normal field E_{e,T}; the flux d_e
  of D_s through the primal edge gives the total normal displacement
  D_n / eps_0 = d_e / (eps_0 L_e) + eps_b E_i,n(m_{e,T}). The segment ADE
  then updates the total E_{e,T} exactly as for the total-field solver, and
  the voltage seen by the scattered-field curl is the *scattered* part,
  v_e = sum_T h_{e,T} (E_{e,T} - E_i,n(m_{e,T})). E_i,n = E_i . n_e with
  n_e = t_e x z the normal of the flux d_e and m_{e,T} the segment midpoint.

Time levels follow the leapfrog: TM evaluates E_i at the integer levels n
(ADE drive) and n + 1 (Debye), TE at n for the voltage and n + 1 for the
segment update (D^{n+1}).

`attach_dispersive_plane_wave` is called by `UnstructuredFIT2D.add_plane_wave`
when the mesh holds dispersive materials.
"""

import numpy as np
from scipy.constants import epsilon_0 as eps_0

from ._core import CapacitiveUpdate

__all__ = ["DelayTable", "attach_dispersive_plane_wave"]


class DelayTable:
    """
    E_i at fixed targets and integer time levels, s(n dt - tau) tabulated.

    The pulse s is sampled once on a grid of step dt / sub and the delays
    tau = (pos - s0) / v are linearly interpolated (error ~ (w dt / sub)^2 / 8),
    as `UnstructuredFIT2D._build_pw_table`.

    Parameters
    ----------
    pw : dict
        The solver's ``plane_wave`` description.
    pos : ndarray
        k.r of the targets [m].
    weight : ndarray
        Per-target factor (amplitude x projection).
    dt : float
    sub : int
    """

    def __init__(self, pw, pos, weight, dt, sub=8):
        tau = np.maximum((np.asarray(pos) - pw["s0"]) / pw["v"] / dt * sub, 0.0)
        self.j = np.floor(tau).astype(int)
        self.f = tau - self.j
        self.w = pw["amp"] * np.asarray(weight, float)
        self.sub, self.dt, self.s = sub, dt, pw["s"]
        self._S = np.zeros(0)

    def __call__(self, n):
        top = self.sub * n
        need = top + 1
        if self._S.size < need:
            m = np.arange(max(2 * self._S.size, need + 4096))
            self._S = np.asarray(self.s(m * self.dt / self.sub), dtype=float)
        i0 = top - self.j
        i1 = i0 - 1
        s0 = np.where(i0 >= 0, self._S[np.maximum(i0, 0)], 0.0)
        s1 = np.where(i1 >= 0, self._S[np.maximum(i1, 0)], 0.0)
        return self.w * ((1 - self.f) * s0 + self.f * s1)


class _ScatteredFieldCapacitiveUpdate(CapacitiveUpdate):
    """`CapacitiveUpdate` whose ADEs are driven by u_s + E_i (TM)."""

    def __init__(self, base, solver, tables):
        self.__dict__.update(base.__dict__)
        self._solver = solver
        self._tables = tables  # one DelayTable per group (node targets)

    def step(self, u, r):
        n = self._solver.n
        acc = self.a1 * u + self.dt * r
        e_n = [t(n) for t in self._tables]
        for g, ei in zip(self.groups, e_n):
            idx, c = g["idx"], g["c"]
            un = u[idx] + ei
            corr = np.zeros(idx.size)
            for k, (alpha, xi, zeta, p, p_old) in enumerate(g["lorentz"]):
                p_new = alpha * p + xi * p_old + zeta * un
                corr += p_new - p
                g["lorentz"][k] = (alpha, xi, zeta, p_new, p)
            for kappa, _, p in g["debye"]:
                corr += (kappa - 1) * p
            np.add.at(acc, idx, -c * corr)
        u_new = acc / self.den
        for g, t, ei in zip(self.groups, self._tables, e_n):
            if not g["debye"]:
                continue
            idx = g["idx"]
            tot = u_new[idx] + t(n + 1) + u[idx] + ei
            for k, (kappa, beta, p) in enumerate(g["debye"]):
                g["debye"][k] = (kappa, beta, kappa * p + beta * tot)
        return u_new


def _attach_tm(solver):
    pw = solver.plane_wave
    nodes = solver.mesh.nodes[solver.free]
    tables = []
    for g in solver._update.groups:
        pos = nodes[g["idx"]] @ pw["k"]
        tables.append(DelayTable(pw, pos, np.ones(len(pos)), solver.dt))
    solver._update = _ScatteredFieldCapacitiveUpdate(solver._update, solver, tables)


def _attach_te(solver):
    pw = solver.plane_wave
    mesh = solver.mesh
    p = np.array([-pw["k"][1], pw["k"][0]])  # E_i direction z x k
    for g in solver._te_groups:
        # segment midpoints: (edge midpoint + circumcentre of T) / 2
        cells = g["cells"]
        mid = 0.5 * (mesh.edge_midpoint[g["e"]] + mesh.cell_circumcenter[cells])
        t = mesh.edge_tangent[g["e"]]
        n_e = np.stack([t[:, 1], -t[:, 0]], 1)
        g["einc"] = DelayTable(pw, mid @ pw["k"], n_e @ p, solver.dt)
    eps_b = pw["eps_b"]

    def te_voltage(d):
        v = solver.stiff * d
        n = solver.n
        for g in solver._te_groups:
            es = g["E"] - g["einc"](n)
            v += np.bincount(g["e"], g["h"] * es, minlength=v.size)
        return v

    def te_ade_step(d):
        n1 = solver.n + 1
        for g in solver._te_groups:
            En = g["E"]
            Dn = d[g["e"]] / (g["L"] * eps_0) + eps_b * g["einc"](n1)
            psum = np.zeros(En.size)
            for k, (alpha, xi, zeta, pp, p_old) in enumerate(g["lorentz"]):
                p_new = alpha * pp + xi * p_old + zeta * En
                psum += p_new
                g["lorentz"][k] = (alpha, xi, zeta, p_new, pp)
            for kappa, beta, pp in g["debye"]:
                psum += kappa * pp + beta * En
            E_new = (Dn - psum) / (g["eps_inf"] + g["beta"])
            for k, (kappa, beta, pp) in enumerate(g["debye"]):
                g["debye"][k] = (kappa, beta, kappa * pp + beta * (E_new + En))
            g["E"] = E_new

    solver._te_voltage = te_voltage
    solver._te_ade_step = te_ade_step


def attach_dispersive_plane_wave(solver):
    """
    Make the scattered-field plane wave of `solver` drive its dispersive
    materials with the total field (see the module docstring). Called by
    `UnstructuredFIT2D.add_plane_wave`.
    """
    if solver.pol == "TM":
        _attach_tm(solver)
    else:
        _attach_te(solver)


def absorbed_power_weights(solver):
    """
    Quadrature of int (-Im eps(w)) |E|^2 dA over the dispersive cells.

    Returns a list of (material, kind, index, weight) with kind 'node'
    (TM: free-node indices and eps_0-free dispersive areas, so that
    sum w |E_n|^2 ~ int_disp |E_z|^2 dA) or 'segment' (TE: segment index into
    the group's E array and h_{e,T} L_e, consistent with the discrete energy
    (1/2) sum v_e d_e).
    """
    out = []
    if solver.pol == "TM":
        for (mat, _), g in zip(solver.dispersive, solver._update.groups):
            out.append((mat, "node", g["idx"], g["c"] / eps_0))
    else:
        for (mat, _), g in zip(solver.dispersive, solver._te_groups):
            out.append((mat, "segment", np.arange(g["e"].size), g["h"] * g["L"]))
    return out
