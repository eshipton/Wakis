# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""Shared helpers of the unstructured FIT solvers (materials, updates, monitors)."""

import numpy as np
import scipy.sparse as sp
from scipy.constants import epsilon_0 as eps_0
from scipy.constants import mu_0
from scipy.sparse.linalg import LinearOperator, eigsh

from ..dispersive import DispersiveMaterial, _ade_coefficients

ETA_0 = np.sqrt(mu_0 / eps_0)


def cell_materials(mesh, materials):
    """
    Per-cell material arrays from a ``{tag: material}`` mapping.

    A material is ``[eps_r, mu_r]``, ``[eps_r, mu_r, sigma]``, a
    `wakis.dispersive.DispersiveMaterial` or the string ``"pec"``. Tags
    missing from `materials` are vacuum.

    Returns
    -------
    eps_r, mu_r, sigma : ndarray (n_cells,)
        Relative permittivity (eps_inf for dispersive cells), relative
        permeability and conductivity [S/m].
    pec : ndarray of bool (n_cells,)
        Perfectly conducting cells.
    dispersive : list of (DispersiveMaterial, ndarray of bool)
        Dispersive materials and the cells they fill.
    """
    nc = mesh.n_cells
    eps_r, mu_r, sigma = np.ones(nc), np.ones(nc), np.zeros(nc)
    pec = np.zeros(nc, bool)
    dispersive = []
    for tag, mat in (materials or {}).items():
        sel = mesh.cell_tags == tag
        if isinstance(mat, str):
            if mat.lower() != "pec":
                raise ValueError(f"unknown material {mat!r}")
            pec |= sel
        elif isinstance(mat, DispersiveMaterial):
            eps_r[sel] = mat.eps_inf
            if mat.poles:
                dispersive.append((mat, sel))
            sigma[sel] = mat.sigma
        else:
            mat = list(mat)
            eps_r[sel] = mat[0]
            mu_r[sel] = mat[1] if len(mat) > 1 else 1.0
            sigma[sel] = mat[2] if len(mat) > 2 else 0.0
    return eps_r, mu_r, sigma, pec, dispersive


class CapacitiveUpdate:
    """
    Semi-implicit, local update of capacitive degrees of freedom.

    Solves, for every dof k (an edge in 3D, a node for 2D TM),

        C0 (u' - u) + G dt (u' + u) / 2 + sum_m c_m sum_poles (p' - p) = dt r

    where C0 [F] is the diagonal capacitance (eps_inf of dispersive cells),
    G [S] a conductance (conductivity and absorbing boundaries), c_m [F] the
    vacuum capacitance of the part of the dual cell filled with dispersive
    material m, p its pole polarizations (same units as u) integrated with
    the ADEs of `wakis.dispersive` and r the curl term minus the source
    current [A]. The update is linear and explicit (no global solve).

    Parameters
    ----------
    C0, G : ndarray
        Capacitance and conductance per dof.
    groups : list of (DispersiveMaterial, ndarray of int, ndarray of float)
        Dispersive material, dof indices and vacuum capacitances c_m [F].
    dt : float
        Timestep [s].
    """

    def __init__(self, C0, G, groups, dt):
        self.dt = dt
        self.groups = []
        a1 = C0 - G * dt / 2
        den = C0 + G * dt / 2
        for mat, idx, c in groups:
            lorentz, debye = _ade_coefficients(mat, dt)
            beta = sum(b for _, b in debye)
            np.add.at(a1, idx, -c * beta)
            np.add.at(den, idx, c * beta)
            self.groups.append(
                {
                    "idx": idx,
                    "c": c,
                    "lorentz": [
                        (*cf, np.zeros(idx.size), np.zeros(idx.size)) for cf in lorentz
                    ],
                    "debye": [(*cf, np.zeros(idx.size)) for cf in debye],
                }
            )
        self.a1, self.den = a1, den

    def step(self, u, r):
        """Return u^{n+1} from u^n and the right-hand side r = curl - source."""
        acc = self.a1 * u + self.dt * r
        for g in self.groups:
            idx, c = g["idx"], g["c"]
            un = u[idx]
            corr = np.zeros(idx.size)
            for k, (alpha, xi, zeta, p, p_old) in enumerate(g["lorentz"]):
                p_new = alpha * p + xi * p_old + zeta * un
                corr += p_new - p
                g["lorentz"][k] = (alpha, xi, zeta, p_new, p)
            for kappa, _, p in g["debye"]:
                corr += (kappa - 1) * p
            np.add.at(acc, idx, -c * corr)
        u_new = acc / self.den
        for g in self.groups:
            idx = g["idx"]
            for k, (kappa, beta, p) in enumerate(g["debye"]):
                g["debye"][k] = (kappa, beta, kappa * p + beta * (u_new[idx] + u[idx]))
        return u_new

    def reset(self):
        for g in self.groups:
            g["lorentz"] = [(a, x, z, 0 * p, 0 * q) for a, x, z, p, q in g["lorentz"]]
            g["debye"] = [(k, b, 0 * p) for k, b, p in g["debye"]]


def check_dispersive_stability(materials, dt, courant):
    """Raise if `dt` is unstable for any dispersive material (von Neumann)."""
    for mat in materials:
        if not mat.is_stable(dt, courant):
            raise ValueError(
                f"dt={dt:.3e} s is unstable for {mat!r}; pass a smaller `dt` "
                "or `cfl` (see DispersiveMaterial.is_stable)."
            )


def max_eigenvalue(K, M_inv_sqrt=None, M_solve=None, n=None):
    """
    Largest eigenvalue of M^-1 K (Lanczos, `scipy.sparse.linalg.eigsh`).

    Parameters
    ----------
    K : sparse matrix
        Symmetric positive semi-definite stiffness.
    M_inv_sqrt : ndarray, optional
        Diagonal M^-1/2 (diagonal mass).
    M_solve : callable, optional
        x -> M^-1 x for a non-diagonal mass (generalized problem).
    """
    n = K.shape[0]
    if M_inv_sqrt is not None:
        op = LinearOperator(
            (n, n), matvec=lambda x: M_inv_sqrt * (K @ (M_inv_sqrt * x)), dtype=float
        )
        return eigsh(op, k=1, which="LA", return_eigenvectors=False, tol=1e-4)[0]
    op = LinearOperator((n, n), matvec=lambda x: M_solve(K @ x), dtype=float)
    from scipy.sparse.linalg import eigs

    return np.max(
        np.real(eigs(op, k=1, which="LR", return_eigenvectors=False, tol=1e-4))
    )


class Recorder:
    """Point probes and running DFTs of a solver's electric-type dofs."""

    def __init__(self):
        self.probes = []  # (weights matrix, list of samples)
        self.dfts = []  # dict(freqs, acc)

    def add_probe(self, P):
        rec = {"P": P, "samples": []}
        self.probes.append(rec)
        return rec

    def add_dft(self, freqs, n):
        freqs = np.atleast_1d(np.asarray(freqs, dtype=float))
        rec = {"freqs": freqs, "data": np.zeros((freqs.size, n), dtype=complex)}
        self.dfts.append(rec)
        return rec

    def record(self, u, t, dt):
        for rec in self.probes:
            rec["samples"].append(rec["P"] @ u)
        for rec in self.dfts:
            # Kernel exp(-i w t) and the dt factor, as wakis.DFTMonitor
            rec["data"] += (
                np.exp(-2j * np.pi * rec["freqs"] * t)[:, None] * u[None] * dt
            )


def diag(x):
    return sp.diags(np.asarray(x, dtype=float))
