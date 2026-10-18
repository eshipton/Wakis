# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

import copy

import numpy as np
from scipy.sparse import csr_matrix

from . import readouts

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


def _host(array):
    return np.asarray(array.get() if hasattr(array, "get") else array)


class _LumpedTranspose:
    """
    Transposed update of one `LumpedElement` on its edges e.

    Forward (`LumpedElement.update`, solver step, `LumpedElement.post_step`)::

        J_L' = J_L + dt gamma E_e          (enters solver.J on e)
        E*   = step(H, E) - dt ieps J
        E'_e = a E_e + b (E*_e - E_e)
    """

    def __init__(self, element, key, dt, ieps):
        self.element = element
        self.key = key
        self.idx = _host(element.flat_idx)
        self.a = _host(element.coef_a)
        self.b = _host(element.coef_b)
        self.gamma = element.gamma
        self.dt = dt
        self.dt_ieps = dt * ieps[self.idx]

    @property
    def keys(self):
        return [] if self.gamma is None else [self.key]

    def get_state(self, state):
        for key in self.keys:
            state[key] = _host(self.element.J_L).copy()

    def set_state(self, state):
        for key in self.keys:
            self.element.J_L = np.array(state[key])
        # The element current already in solver.J is reset together with J
        self.element.J_old = np.zeros_like(self.element.J_old)

    def transpose_post(self, lam, lam_star, extra, out):
        lam_e = lam["E"][self.idx]
        lam_star[self.idx] = self.b * lam_e
        extra[self.idx] += (self.a - self.b) * lam_e

    def transpose_pre(self, lam, lam_star, out):
        if self.gamma is None:
            return
        lam_J = lam[self.key] - self.dt_ieps * lam_star[self.idx]
        out[self.key] = lam_J
        out["E"][self.idx] += self.dt * self.gamma * lam_J


class _DispersiveTranspose:
    """
    Transposed ADE correction of one dispersive group (`_one_step_dispersive`)
    on its edges e, with E_n = E^n_e and E* the solver step::

        acc  = (E*_e - E_n) eps_s + a1 E_n - w sum_L (p'_L - p_L)
               - w sum_D (kappa - 1) p_D
        p'_L = alpha p_L + xi p_L,old + zeta E_n,   p'_L,old = p_L
        E'_e = acc / den,   p'_D = kappa p_D + beta (E'_e + E_n)
    """

    def __init__(self, group, gi):
        self.group = group
        self.idx = group["idx_host"]
        self.w = _host(group["weight"])
        self.eps_s = _host(group["eps_s"])
        self.a1 = _host(group["a1"])
        self.den = _host(group["den"])
        self.lorentz = [cf[:3] for cf in group["lorentz"]]
        self.debye = [cf[:2] for cf in group["debye"]]
        self.lkeys = [
            (f"pL{gi}_{k}", f"pLold{gi}_{k}") for k in range(len(self.lorentz))
        ]
        self.dkeys = [f"pD{gi}_{k}" for k in range(len(self.debye))]
        self.lam_acc = None

    @property
    def keys(self):
        return [k for pair in self.lkeys for k in pair] + self.dkeys

    def get_state(self, state):
        g = self.group
        for (kp, ko), cf in zip(self.lkeys, g["lorentz"]):
            state[kp], state[ko] = _host(cf[3]).copy(), _host(cf[4]).copy()
        for kd, cf in zip(self.dkeys, g["debye"]):
            state[kd] = _host(cf[2]).copy()

    def set_state(self, state):
        g = self.group
        g["lorentz"] = [
            (*cf[:3], np.array(state[kp]), np.array(state[ko]))
            for (kp, ko), cf in zip(self.lkeys, g["lorentz"])
        ]
        g["debye"] = [
            (*cf[:2], np.array(state[kd])) for kd, cf in zip(self.dkeys, g["debye"])
        ]

    def transpose_post(self, lam, lam_star, extra, out):
        # E'_e feeds the new E and the new Debye polarizations
        lam_new = lam["E"][self.idx]
        for kd, (kappa, beta) in zip(self.dkeys, self.debye):
            lam_new = lam_new + beta * lam[kd]
        lam_acc = lam_new / self.den
        lam_n = (self.a1 - self.eps_s) * lam_acc
        for kd, (kappa, beta) in zip(self.dkeys, self.debye):
            out[kd] = kappa * lam[kd] - self.w * (kappa - 1) * lam_acc
            lam_n = lam_n + beta * lam[kd]
        for (kp, ko), (alpha, xi, zeta) in zip(self.lkeys, self.lorentz):
            lam_p_new = lam[kp] - self.w * lam_acc
            out[kp] = alpha * lam_p_new + lam[ko] + self.w * lam_acc
            out[ko] = xi * lam_p_new
            lam_n = lam_n + zeta * lam_p_new
        lam_star[self.idx] = self.eps_s * lam_acc
        extra[self.idx] += lam_n
        self.lam_acc = lam_acc


class AdjointStepper:
    def __init__(self, solver, elements=()):
        """
        Exact (Hermitian) transpose of one `SolverFIT3D` time step, for
        adjoint runs.

        Without sources the step is linear in the state
        x = (H, E[, psi][, J_L][, p]) (fields, CPML auxiliary variables,
        lumped inductor currents, dispersive polarizations), x' = U x. `step`
        applies U^H (= U^T for real fields) to the adjoint variables
        (Lagrange multipliers) of x', giving those of x.

        Standard step, with A = ``tDsiDmuiDaC``, B = ``itDaiDepsDstC`` and
        S = diag(dt sigma ieps) when conductivity is enabled::

            H' = H - dt A E,   E* = (1 - S) E + dt B H' - dt ieps J
            lam_H = lam_H' + dt B^H lam_E*,
            lam_E = (1 - S) lam_E* - dt A^H lam_H

        CPML adds the psi variables to the state. Lumped elements and
        dispersive groups then correct E on their own edges,
        E' = P(E*, E, aux); their transposes give lam_E* from lam_E' and add
        terms to lam_E and to the auxiliary multipliers. See the physics guide
        for all transposed updates.

        Parameters
        ----------
        solver : SolverFIT3D
            Solver after its first time step (operators in final form).
        elements : sequence of LumpedElement, optional
            Lumped elements and ports of the run (objects with `post_step`),
            after their first update.

        Raises
        ------
        NotImplementedError
            For GPU, MPI, ABC, PML, SIBC, active TFSF injection, or lumped
            elements sharing edges with each other or with dispersive media.

        Notes
        -----
        Impressed currents J and M (`CurrentSource`, `MagneticCurrentSource`,
        `ModeSource`, the source of a `LumpedPort`) do not depend on the state
        and do not enter the transposed step.
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
        if solver.source_type == "tfsf" and not solver.injection_done:
            raise NotImplementedError("The adjoint does not support TFSF injection.")

        self.solver = solver
        self.N = solver.N
        self.dt = solver.dt
        self.cpml = solver.activate_cpml
        self.dtype = solver.E.array.dtype
        self.ieps = solver.ieps.toarray().copy()
        self.imu = solver.imu.toarray().copy()
        self.loss = None
        if solver.use_conductivity:
            self.loss = 1.0 - self.dt * solver.sigma.toarray() * self.ieps
        self.lam_star = None

        self.lumped = [
            _LumpedTranspose(el, f"JL{i}", self.dt, self.ieps)
            for i, el in enumerate(elements)
        ]
        self.dispersive = [
            _DispersiveTranspose(g, gi)
            for gi, g in enumerate(getattr(solver, "_dispersive_groups", []))
        ]
        corrected = [ext.idx for ext in self.lumped + self.dispersive]
        if corrected:
            joined = np.concatenate(corrected)
            if np.unique(joined).size != joined.size:
                raise NotImplementedError(
                    "Lumped elements must not share edges with each other or "
                    "with dispersive media in the adjoint."
                )

        if not self.cpml:
            # Hermitian transposes conjugate the Bloch phases of the curls
            self.At = csr_matrix(solver.tDsiDmuiDaC).conj().T.tocsr()
            self.Bt = csr_matrix(solver.itDaiDepsDstC).conj().T.tocsr()
            self.psi_names = []
            return

        self.ops_t = {
            term[4]: csr_matrix(getattr(solver, term[4])).conj().T.tocsr()
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
        """Zero adjoint state: dict with 'H', 'E', CPML psi and auxiliary arrays."""
        state = {k: np.zeros_like(v) for k, v in self.get_state().items()}
        state["H"] = np.zeros(3 * self.N, dtype=self.dtype)
        state["E"] = np.zeros(3 * self.N, dtype=self.dtype)
        return state

    def get_state(self):
        """Copy of the solver's forward state (fields, psi, auxiliary arrays)."""
        state = {
            "H": self.solver.H.toarray().copy(),
            "E": self.solver.E.toarray().copy(),
        }
        for name in self.psi_names:
            state[name] = getattr(self.solver, name).copy()
        for ext in self.lumped + self.dispersive:
            ext.get_state(state)
        return state

    def set_state(self, state):
        """
        Overwrite the solver's forward state with the arrays in `state`.

        The impressed and conduction currents (solver.J, solver.M and their
        bookkeeping) are reset to zero, so that the next forward step applies
        the homogeneous step U to `state`.
        """
        solver = self.solver
        solver.H.fromarray(state["H"])
        solver.E.fromarray(state["E"])
        solver.J.fromarray(np.zeros(3 * self.N, dtype=self.dtype))
        if solver.M is not None:
            solver.M.fromarray(np.zeros(3 * self.N, dtype=self.dtype))
        if hasattr(solver, "J_old"):
            solver.J_old = np.zeros(3 * self.N, dtype=self.dtype)
        for name in self.psi_names:
            setattr(solver, name, state[name].copy())
        for ext in self.lumped + self.dispersive:
            ext.set_state(state)

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
            Adjoint variables of the state before that step. The adjoint
            variable of the E computed by the solver step itself (before
            lumped and dispersive corrections) is kept in `lam_star`.
        """
        lam_star = lam["E"].copy()
        extra = np.zeros_like(lam_star)
        out = {}
        for ext in self.lumped + self.dispersive:
            ext.transpose_post(lam, lam_star, extra, out)
        out.update(self._base_step(lam, lam_star))
        for ext in self.lumped:
            ext.transpose_pre(lam, lam_star, out)
        if self.lumped or self.dispersive:
            out["E"] = out["E"] + extra
        self.lam_star = lam_star
        return out

    def _base_step(self, lam, lam_E):
        """Transpose of the solver step, given the adjoint lam_E of its E."""
        if not self.cpml:
            lam_H = lam["H"] + self.dt * (self.Bt @ lam_E)
            if self.loss is not None:
                lam_E = self.loss * lam_E
            return {"H": lam_H, "E": lam_E - self.dt * (self.At @ lam_H)}

        out = {}
        # Reverse of the E half step: E' = (1 - S) E + dt ieps s (Dt H' + psiE')
        nu = self.dt * self.ieps * lam_E
        lam_H = lam["H"].copy()
        for T, d, S, s, op, prefix in _E_TERMS:
            nu_T = self._comp(nu, T)
            beta = s * nu_T
            for name, b, c, idx in self.faces[op]:
                mu = lam[name] + s * nu_T[idx]
                beta[idx] += c * mu
                out[name] = b * mu
            self._comp(lam_H, S)[:] += self.ops_t[op] @ beta
        lam_E = lam_E.copy() if self.loss is None else self.loss * lam_E

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
        of `design` set from the density rho and returns the raw DFT outputs
        of the monitors. `vjp(G)` runs the transposed time steps backwards and
        returns dF/drho for an objective F of those outputs, given its
        sensitivities G = dF/dRe(X) + 1j dF/dIm(X).

        The gradient uses that ieps enters each step only as a factor of the
        increment of the solver step, q^n = E*^{n+1} - E^n
        = dt ieps (curl H - J), so dF/dieps = Re sum_n conj(lam_E*^{n+1})
        q^n / ieps on the design edges, with lam_E* the adjoint of the E
        computed by the solver step. q^n/ieps is stored on those edges during
        the forward run.

        Parameters
        ----------
        make_solver : callable
            Function without arguments returning a fresh `SolverFIT3D`
            (possibly with dispersive materials). The time step dt must not
            depend on the design.
        sources : list
            Sources (`CurrentSource`, `MagneticCurrentSource`, `ModeSource`)
            and lumped elements/ports, independent of the design. They are
            deep-copied before each run, together with the monitors.
        monitors : dict of str to monitor
            Named `DFTMonitor` (E/H components), `FluxMonitor`,
            `Near2FarMonitor`, `ModeMonitor` or `LumpedReadout`; see
            `readouts.outputs` for the raw outputs of each.
        design : MaterialGrid
            Design region. It must not share edges with lumped elements,
            dispersive media (other than its own), mode source planes or
            mode monitor planes.
        Nt : int
            Number of time steps.
        store_dtype : numpy dtype, optional
            Precision of the stored increments (Nt x N_design values; complex
            for complex fields). float32 halves the memory at ~1e-7 relative
            gradient error.
        """
        for name, monitor in monitors.items():
            readouts.check(name, monitor)
        self.make_solver = make_solver
        self.sources = list(sources)
        self.monitors = dict(monitors)
        self.design = design
        self.Nt = int(Nt)
        self.store_dtype = store_dtype
        self.rho = None
        self.readouts = None

    def output_shapes(self):
        """
        Shapes of the arrays returned by `forward`, without time-stepping.

        Also stores in `readouts` copies of the monitors initialized on a
        fresh solver (sample geometry and weights), for the helper functions
        that turn raw outputs into flux, mode amplitudes or far fields.

        Returns
        -------
        dict
            {monitor name: {output key: shape}}.
        """
        solver = self.make_solver()
        self.readouts = {}
        shapes = {}
        for name, template in self.monitors.items():
            monitor = copy.deepcopy(template)
            readouts.initialize(monitor, solver)
            self.readouts[name] = monitor
            shapes[name] = readouts.shapes(monitor)
        return shapes

    def forward(self, rho):
        """
        Run the simulation for density rho.

        The time loop is that of `SolverFIT3D.emsolve`: source and element
        updates, `one_step`, element `post_step`, monitor updates.

        Parameters
        ----------
        rho : ndarray
            Density on the design grid, shape `design.shape`.

        Returns
        -------
        dict
            {monitor name: {output key: complex ndarray}}, see
            `readouts.outputs`. A `DFTMonitor` gives its `get_field` arrays
            [field units x s].
        """
        rho = np.array(rho, dtype=float)
        solver = self.make_solver()
        self.design.apply(solver, rho)
        sources, monitors = copy.deepcopy((self.sources, self.monitors))
        elements = [src for src in sources if hasattr(src, "post_step")]
        for name, monitor in monitors.items():
            if isinstance(monitor, readouts.LumpedReadout) and not any(
                monitor.element is el for el in elements
            ):
                raise ValueError(f"The element of readout {name!r} is not a source.")

        idx = self.design.flat_idx
        complex_fields = np.iscomplexobj(solver.E.array)
        store = np.result_type(self.store_dtype, np.complex64 if complex_fields else 1)
        ieps_d = solver.ieps.array[idx].copy()
        increments = np.empty((self.Nt, idx.size), dtype=store)
        disp = self.design.dispersive_group(solver)
        if disp is not None:
            ade_terms = np.empty((self.Nt, idx.size), dtype=store)
        for n in range(self.Nt):
            t = n * solver.dt
            for src in sources:
                src.update(solver, t)
            e_old = solver.E.array[idx]
            if disp is not None:
                state_old = self.design.ade_state(disp, solver)
            if disp is None:
                solver.one_step()
                increments[n] = (solver.E.array[idx] - e_old) / ieps_d
            else:
                # The ADE overwrites the design edges: record the parameter
                # sensitivity of the correction instead (see MaterialGrid)
                solver.one_step()
                ade_terms[n] = self.design.ade_sensitivity(
                    disp, solver, e_old, state_old
                )
            for el in elements:
                el.post_step(solver, t)
            for monitor in monitors.values():
                monitor.update(solver, t)

        self._check_overlaps(solver, elements, monitors)
        self._solver, self._monitors, self._elements = solver, monitors, elements
        self._increments = increments if disp is None else ade_terms
        self.rho = rho
        return {name: readouts.outputs(m) for name, m in monitors.items()}

    def _check_overlaps(self, solver, elements, monitors):
        """Raise if the design shares edges with design-independent models."""
        idx = self.design.flat_idx
        own = self.design.dispersive_group(solver)
        regions = [_host(el.flat_idx) for el in elements]
        regions += [
            g["idx_host"]
            for g in getattr(solver, "_dispersive_groups", [])
            if g is not own
        ]
        if np.any(np.isin(idx, np.concatenate(regions or [[]]).astype(int))):
            raise NotImplementedError(
                "The design region overlaps a lumped element or dispersive medium."
            )

    def vjp(self, cotangents):
        """
        Gradient dF/drho by an adjoint run, after `forward`.

        Parameters
        ----------
        cotangents : dict
            Same structure as the output of `forward` (missing entries are
            zero), holding G = dF/dRe(X) + 1j dF/dIm(X) for each output X.

        Returns
        -------
        ndarray
            dF/drho with shape `design.shape`.
        """
        if self.rho is None:
            raise RuntimeError("Call forward before vjp.")
        solver = self._solver
        stepper = AdjointStepper(solver, self._elements)
        dt = solver.dt
        complex_fields = np.iscomplexobj(solver.E.array)

        keys = {id(ext.element): ext.key for ext in stepper.lumped}
        taps = []
        for name, comps in cotangents.items():
            taps += readouts.taps(self._monitors[name], solver, comps, keys)

        idx = self.design.flat_idx
        own = self.design.dispersive_group(solver)
        ade = None
        if own is not None:
            ade = next(ext for ext in stepper.dispersive if ext.group is own)
        grad = np.zeros(idx.size)
        lam = stepper.zero_state()
        for n in reversed(range(self.Nt)):
            t = n * dt
            # A DFT sample X = sum_n w_n W x^{n+1} gives lam += W^H conj(w_n) G
            for tap in taps:
                if tap.skip_last and n == self.Nt - 1:
                    continue
                w_conj = dt * np.exp(2j * np.pi * tap.freqs * (t + tap.offset))
                source = tap.WH @ (w_conj @ tap.G)
                lam[tap.field] += source if complex_fields else source.real
            lam = stepper.step(lam)
            sens = stepper.lam_star[idx] if ade is None else ade.lam_acc
            grad += np.real(np.conj(sens) * self._increments[n])
        return self.design.vjp(self.rho, grad)
