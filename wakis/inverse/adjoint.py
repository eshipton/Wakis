# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

import copy

import numpy as np
from scipy.sparse import csr_matrix

from ..field_monitors import DFTMonitor

# CPML curl terms (target component T, derivative axis d, source component S,
# sign s, operator, auxiliary-variable prefix), read off `_one_step_cpml`:
#   H_T -= dt imu_T s (D E_S + psiH)   and   E_T += dt ieps_T s (Dt H_S + psiE)
# where psi lives on the CPML faces normal to d.
_H_TERMS = (
    ("x", "y", "z", 1.0, "dxy", "psiHa_x"),
    ("x", "z", "y", -1.0, "dxz", "psiHb_x"),
    ("y", "z", "x", 1.0, "dyz", "psiHa_y"),
    ("y", "x", "z", -1.0, "dyx", "psiHb_y"),
    ("z", "x", "y", 1.0, "dzx", "psiHa_z"),
    ("z", "y", "x", -1.0, "dzy", "psiHb_z"),
)
_E_TERMS = (
    ("x", "y", "z", 1.0, "dtxy", "psiEa_x"),
    ("x", "z", "y", -1.0, "dtxz", "psiEb_x"),
    ("y", "z", "x", 1.0, "dtyz", "psiEa_y"),
    ("y", "x", "z", -1.0, "dtyx", "psiEb_y"),
    ("z", "x", "y", 1.0, "dtzx", "psiEa_z"),
    ("z", "y", "x", -1.0, "dtzy", "psiEb_z"),
)


class AdjointStepper:
    def __init__(self, solver):
        """
        Exact transpose of one `SolverFIT3D` time step, for adjoint runs.

        Without sources the leapfrog step is linear in the state
        x = (H, E[, psi]), x' = U x. `step` applies U^T to the adjoint
        variables (Lagrange multipliers) of x', giving those of x.

        Standard step, with A = ``tDsiDmuiDaC``, B = ``itDaiDepsDstC`` and
        S = diag(dt sigma ieps) when conductivity is enabled::

            H' = H - dt A E,  E' = (1 - S) E + dt B H'
            lam_H = lam_H' + dt B^T lam_E',  lam_E = (1 - S) lam_E' - dt A^T lam_H

        CPML step: the auxiliary variables psi are part of the state; see the
        physics guide for the transposed sweep.

        Parameters
        ----------
        solver : SolverFIT3D
            Solver after its first time step (operators in final form).

        Raises
        ------
        NotImplementedError
            For GPU, MPI, ABC, PML, SIBC, or active TFSF injection.
        """
        if solver.use_gpu:
            raise NotImplementedError("The adjoint runs on CPU only.")
        if solver.use_mpi:
            raise NotImplementedError("The adjoint does not support MPI.")
        if solver.activate_abc:
            raise NotImplementedError("The adjoint does not support ABC (Mur).")
        if solver.activate_pml:
            raise NotImplementedError("The adjoint does not support PML; use CPML.")
        if solver.use_sibc:
            raise NotImplementedError("The adjoint does not support SIBC.")
        if np.iscomplexobj(solver.E.array):
            # Complex (Bloch) fields need Hermitian transposes and complex
            # parameter sensitivities, not yet implemented
            raise NotImplementedError("The adjoint does not support complex fields.")
        if solver.source_type == "tfsf" and not solver.injection_done:
            raise NotImplementedError("The adjoint does not support TFSF injection.")

        self.solver = solver
        self.N = solver.N
        self.dt = solver.dt
        self.cpml = solver.activate_cpml
        self.ieps = solver.ieps.toarray().copy()
        self.imu = solver.imu.toarray().copy()
        self.loss = None
        if solver.use_conductivity:
            self.loss = 1.0 - self.dt * solver.sigma.toarray() * self.ieps

        if not self.cpml:
            self.At = csr_matrix(solver.tDsiDmuiDaC).T.tocsr()
            self.Bt = csr_matrix(solver.itDaiDepsDstC).T.tocsr()
            self.psi_names = []
            return

        self.ops_t = {
            term[4]: csr_matrix(getattr(solver, term[4])).T.tocsr()
            for term in _H_TERMS + _E_TERMS
        }
        # Faces (prefix suffix, b, c, idx) of each term, in forward order
        self.faces = {}
        for terms, field in ((_H_TERMS, "H"), (_E_TERMS, "E")):
            for T, d, S, s, op, prefix in terms:
                faces = []
                for side, bcs in (("low", solver.bc_low), ("high", solver.bc_high)):
                    if bcs["xyz".index(d)].lower() != "cpml":
                        continue
                    faces.append(
                        (
                            f"{prefix}_{side}",
                            getattr(solver, f"pml_b_{field}_{d}_{side}"),
                            getattr(solver, f"pml_c_{field}_{d}_{side}"),
                            getattr(solver, f"idx_{d}_{side}"),
                        )
                    )
                self.faces[op] = faces
        self.psi_names = [f[0] for faces in self.faces.values() for f in faces]

    def _comp(self, array, c):
        i = "xyz".index(c)
        return array[i * self.N : (i + 1) * self.N]

    def zero_state(self):
        """Zero adjoint state: dict with 'H', 'E' and the CPML psi arrays."""
        state = {
            "H": np.zeros(3 * self.N, dtype=self.ieps.dtype),
            "E": np.zeros(3 * self.N, dtype=self.ieps.dtype),
        }
        for name in self.psi_names:
            state[name] = np.zeros_like(getattr(self.solver, name))
        return state

    def get_state(self):
        """Copy of the solver's forward state (H, E and CPML psi arrays)."""
        state = {
            "H": self.solver.H.toarray().copy(),
            "E": self.solver.E.toarray().copy(),
        }
        for name in self.psi_names:
            state[name] = getattr(self.solver, name).copy()
        return state

    def set_state(self, state):
        """Overwrite the solver's forward state with the arrays in `state`."""
        self.solver.H.fromarray(state["H"])
        self.solver.E.fromarray(state["E"])
        for name in self.psi_names:
            setattr(self.solver, name, state[name].copy())

    def step(self, lam):
        """
        Apply the transposed time step to adjoint variables.

        Parameters
        ----------
        lam : dict of ndarray
            Adjoint variables of the state after a step (keys as in
            `zero_state`).

        Returns
        -------
        dict of ndarray
            Adjoint variables of the state before that step.
        """
        if not self.cpml:
            lam_H = lam["H"] + self.dt * (self.Bt @ lam["E"])
            lam_E = lam["E"] if self.loss is None else self.loss * lam["E"]
            return {"H": lam_H, "E": lam_E - self.dt * (self.At @ lam_H)}

        out = {}
        # Reverse of the E half step: E' = (1 - S) E + dt ieps s (Dt H' + psiE')
        nu = self.dt * self.ieps * lam["E"]
        lam_H = lam["H"].copy()
        for T, d, S, s, op, prefix in _E_TERMS:
            nu_T = self._comp(nu, T)
            beta = s * nu_T
            for name, b, c, idx in self.faces[op]:
                mu = lam[name] + s * nu_T[idx]
                beta[idx] += c * mu
                out[name] = b * mu
            self._comp(lam_H, S)[:] += self.ops_t[op] @ beta
        lam_E = lam["E"].copy() if self.loss is None else self.loss * lam["E"]

        # Reverse of the H half step: H' = H - dt imu s (D E + psiH')
        kappa = -self.dt * self.imu * lam_H
        for T, d, S, s, op, prefix in _H_TERMS:
            kappa_T = self._comp(kappa, T)
            alpha = s * kappa_T
            for name, b, c, idx in self.faces[op]:
                mu = lam[name] + s * kappa_T[idx]
                alpha[idx] += c * mu
                out[name] = b * mu
            self._comp(lam_E, S)[:] += self.ops_t[op] @ alpha
        out["H"] = lam_H
        out["E"] = lam_E
        return out


class AdjointSimulation:
    def __init__(
        self, make_solver, sources, monitors, design, Nt, store_dtype=np.float64
    ):
        """
        Forward and exact discrete adjoint runs for gradient-based design.

        `forward(rho)` runs the time-domain simulation with the permittivity
        of `design` set from the density rho and returns the DFT monitor
        fields. `vjp(G)` runs the transposed time steps backwards and returns
        dF/drho for an objective F of those fields, given its sensitivities
        G = dF/dRe(field) + 1j dF/dIm(field).

        The gradient uses that ieps enters each step only as a factor of the
        E increment, q^n = E^{n+1} - E^n = dt ieps (curl H - J), so
        dF/dieps = sum_n lam_E^{n+1} q^n / ieps on the design edges. q^n/ieps
        is stored on those edges during the forward run.

        Parameters
        ----------
        make_solver : callable
            Function without arguments returning a fresh `SolverFIT3D`. The
            time step dt must not depend on the design.
        sources : list
            Source objects (e.g. `CurrentSource`) independent of the design.
            They are deep-copied before each run.
        monitors : dict of str to DFTMonitor
            Named monitors with E and/or H components, deep-copied before
            each run.
        design : MaterialGrid
            Design region.
        Nt : int
            Number of time steps.
        store_dtype : numpy dtype, optional
            Precision of the stored E increments (Nt x N_design values).
            float32 halves the memory at ~1e-7 relative gradient error.
        """
        for name, monitor in monitors.items():
            if not isinstance(monitor, DFTMonitor):
                raise NotImplementedError(f"Monitor {name!r} must be a DFTMonitor.")
            if any(comp[0] == "J" for comp in monitor.components):
                raise NotImplementedError("J monitors are not differentiable yet.")
        self.make_solver = make_solver
        self.sources = list(sources)
        self.monitors = dict(monitors)
        self.design = design
        self.Nt = int(Nt)
        self.store_dtype = store_dtype
        self.rho = None

    def output_shapes(self):
        """
        Shapes of the arrays returned by `forward`, without time-stepping.

        Returns
        -------
        dict
            {monitor name: {component: (n_freq, nx, ny, nz)}}.
        """
        solver = self.make_solver()
        shapes = {}
        for name, template in self.monitors.items():
            monitor = copy.deepcopy(template)
            monitor._initialize(solver)
            shapes[name] = {
                comp: (monitor.frequencies.size, *monitor.shape[comp])
                for comp in monitor.components
            }
        return shapes

    def forward(self, rho):
        """
        Run the simulation for density rho.

        Parameters
        ----------
        rho : ndarray
            Density on the design grid, shape `design.shape`.

        Returns
        -------
        dict
            {monitor name: {component: complex ndarray (n_freq, nx, ny, nz)}},
            the `DFTMonitor.get_field` arrays [field units x s].
        """
        rho = np.array(rho, dtype=float)
        solver = self.make_solver()
        self.design.apply(solver, rho)
        sources = copy.deepcopy(self.sources)
        monitors = {k: copy.deepcopy(m) for k, m in self.monitors.items()}

        idx = self.design.flat_idx
        ieps_d = solver.ieps.array[idx].copy()
        increments = np.empty((self.Nt, idx.size), dtype=self.store_dtype)
        for n in range(self.Nt):
            t = n * solver.dt
            for src in sources:
                src.update(solver, t)
            e_old = solver.E.array[idx]
            solver.one_step()
            increments[n] = (solver.E.array[idx] - e_old) / ieps_d
            for monitor in monitors.values():
                monitor.update(solver, t)

        self._solver, self._monitors = solver, monitors
        self._increments = increments
        self.rho = rho
        return {
            name: {comp: m.get_field(comp)[0] for comp in m.components}
            for name, m in monitors.items()
        }

    def vjp(self, cotangents):
        """
        Gradient dF/drho by an adjoint run, after `forward`.

        Parameters
        ----------
        cotangents : dict
            Same structure as the output of `forward` (missing entries are
            zero), holding G = dF/dRe(X) + 1j dF/dIm(X) for each field X.

        Returns
        -------
        ndarray
            dF/drho with shape `design.shape`.
        """
        if self.rho is None:
            raise RuntimeError("Call forward before vjp.")
        solver = self._solver
        stepper = AdjointStepper(solver)
        dt = solver.dt

        # A DFT sample X_k = sum_n w_n F^n_k makes F^n_k feel Re(w_n conj(G_k))
        sens = []
        for name, comps in cotangents.items():
            monitor = self._monitors[name]
            for comp, G in comps.items():
                G = np.asarray(G).reshape(monitor.frequencies.size, -1)
                offset = dt if comp[0] == "E" else dt / 2
                sens.append(
                    (
                        comp[0],
                        monitor._flat[comp],
                        np.conj(G),
                        monitor.frequencies,
                        offset,
                    )
                )

        idx = self.design.flat_idx
        grad_ieps = np.zeros(idx.size)
        lam = stepper.zero_state()
        for n in reversed(range(self.Nt)):
            t = n * dt
            for field, flat, conj_G, freqs, offset in sens:
                w = dt * np.exp(-2j * np.pi * freqs * (t + offset))
                lam[field][flat] += np.real(w @ conj_G)
            grad_ieps += lam["E"][idx] * self._increments[n]
            lam = stepper.step(lam)
        return self.design.vjp(self.rho, grad_ieps)
