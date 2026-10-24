# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

import copy

import numpy as np
from scipy.constants import epsilon_0 as eps_0
from scipy.sparse import csr_matrix

from ..dispersive import _ade_coefficients
from . import memory as mem
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


def _hermitian(op, solver):
    """
    Hermitian transpose of a sparse solver operator, as CSR on the solver's
    backend (CuPy on the GPU, MKL when the solver steps with MKL).
    """
    if solver.use_gpu:
        from cupyx.scipy.sparse import csr_matrix as gpu_csr_matrix

        return gpu_csr_matrix(op).conj().T.tocsr()
    op_t = csr_matrix(op).conj().T.tocsr()
    if solver.imported_mkl:
        from sparse_dot_mkl import csr_matrix as mkl_csr_matrix

        return mkl_csr_matrix(op_t)
    return op_t


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
        self.idx = element.flat_idx
        self.idx_host = _host(element.flat_idx)
        self.a = element.coef_a
        self.b = element.coef_b
        self.gamma = element.gamma
        self.dt = dt
        self.dt_ieps = dt * ieps[self.idx]

    @property
    def keys(self):
        return [] if self.gamma is None else [self.key]

    def get_state(self, state):
        for key in self.keys:
            state[key] = self.element.J_L.copy()

    def set_state(self, state, xp):
        for key in self.keys:
            self.element.J_L = xp.array(state[key])
        # The element current already in solver.J is reset together with J
        self.element.J_old = xp.zeros_like(self.element.J_old)

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
    Transposed ADE correction of one dispersive group (`_one_step_dispersive`
    for E or `_update_dispersive_H` for H) on its edges e, with E_n = E^n_e
    and E* the solver step::

        acc  = (E*_e - E_n) eps_s + a1 E_n - w sum_L (p'_L - p_L)
               - w sum_D (kappa - 1) p_D
        p'_L = alpha p_L + xi p_L,old + zeta E_n,   p'_L,old = p_L
        E'_e = acc / den,   p'_D = kappa p_D + beta (E'_e + E_n)
    """

    def __init__(self, group, gi, field="E", prefix="p", s_key="eps_s"):
        self.group = group
        self.field = field
        self.idx = group["idx"]
        self.idx_host = group["idx_host"]
        self.w = group["weight"]
        self.eps_s = group[s_key]
        self.a1 = group["a1"]
        self.den = group["den"]
        self.lorentz = [cf[:3] for cf in group["lorentz"]]
        self.debye = [cf[:2] for cf in group["debye"]]
        self.lkeys = [
            (f"{prefix}L{gi}_{k}", f"{prefix}Lold{gi}_{k}")
            for k in range(len(self.lorentz))
        ]
        self.dkeys = [f"{prefix}D{gi}_{k}" for k in range(len(self.debye))]
        self.lam_acc = None

    @property
    def keys(self):
        return [k for pair in self.lkeys for k in pair] + self.dkeys

    def get_state(self, state):
        g = self.group
        for (kp, ko), cf in zip(self.lkeys, g["lorentz"]):
            state[kp], state[ko] = cf[3].copy(), cf[4].copy()
        for kd, cf in zip(self.dkeys, g["debye"]):
            state[kd] = cf[2].copy()

    def set_state(self, state, xp):
        g = self.group
        g["lorentz"] = [
            (*cf[:3], xp.array(state[kp]), xp.array(state[ko]))
            for (kp, ko), cf in zip(self.lkeys, g["lorentz"])
        ]
        g["debye"] = [
            (*cf[:2], xp.array(state[kd])) for kd, cf in zip(self.dkeys, g["debye"])
        ]

    def transpose_post(self, lam, lam_star, extra, out):
        # E'_e (or H'_e) feeds the new field and the new Debye auxiliaries
        lam_new = lam[self.field][self.idx]
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


class _TensorTranspose:
    """
    Transposed tensor-averaging correction (`wakis.tensor_averaging`) on
    the coupled edges R, with E* the solver step and E^n the old field::

        E'_R = E*_R + K' (E*_R - E^n_R)
        lam_E*_R = (I + K'^H) lam_E'_R,   lam_E^n_R += -K'^H lam_E'_R
    """

    def __init__(self, op, solver):
        self.idx = op.rows_dev
        self.idx_host = op.rows
        self.KpH = _hermitian(op.Kp_host, solver)

    keys = ()

    def get_state(self, state):
        pass

    def set_state(self, state, xp):
        pass

    def transpose_post(self, lam, lam_star, extra, out):
        lam_e = lam["E"][self.idx]
        v = self.KpH @ lam_e
        lam_star[self.idx] = lam_e + v
        extra[self.idx] -= v


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

        CPML adds the psi variables to the state. Lumped elements,
        dispersive groups and the tensor averaging then correct E on their
        own edges,
        E' = P(E*, E, aux); their transposes give lam_E* from lam_E' and add
        terms to lam_E and to the auxiliary multipliers. See the physics guide
        for all transposed updates.

        The adjoint variables live on the solver's device (NumPy arrays on
        CPU, CuPy arrays with ``use_gpu``). With MPI, each rank holds the
        adjoint variables of its z-slab, and the ghost-layer exchanges of the
        forward step are transposed in reverse order (see `step`).

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
            For ABC, PML, SIBC, active TFSF injection, lumped elements
            sharing edges with each other or with dispersive media, and, with
            MPI, for CPML, lumped elements and dispersive media.

        Notes
        -----
        Impressed currents J and M (`CurrentSource`, `MagneticCurrentSource`,
        `ModeSource`, the source of a `LumpedPort`) do not depend on the state
        and do not enter the transposed step.
        """
        if solver.activate_abc:
            raise NotImplementedError("The adjoint does not support ABC (Mur).")
        if solver.activate_pml:
            raise NotImplementedError("The adjoint does not support PML; use CPML.")
        if solver.use_sibc:
            raise NotImplementedError("The adjoint does not support SIBC.")
        if solver.source_type == "tfsf" and not solver.injection_done:
            raise NotImplementedError("The adjoint does not support TFSF injection.")
        self.mpi = solver.use_mpi and solver.grid.use_mpi
        dispersive = getattr(solver, "_dispersive_groups", [])
        dispersive_h = getattr(solver, "_dispersive_h_groups", [])
        tensor = getattr(solver, "tensor_operator", None)
        if tensor is not None and tensor.rows.size == 0:
            tensor = None
        network = getattr(solver, "dispersive_network", None)
        dispersive = dispersive or ([] if network is None else [network])
        from ..beam_fluid import BeamTranspose, LinearizedBeam

        lumped_elements = [
            el for el in elements if not isinstance(el, LinearizedBeam)
        ]
        beam_objs = list(getattr(solver, "_beams", []))
        for el in elements:
            if isinstance(el, LinearizedBeam) and el not in beam_objs:
                beam_objs.append(el)
        self.tete = None
        if getattr(solver, "scheme", "yee") == "tete":
            if lumped_elements or dispersive or dispersive_h or tensor or beam_objs:
                raise NotImplementedError(
                    "The TE/TM adjoint step supports no lumped elements, "
                    "dispersive media, tensor averaging or linearized beams."
                )
            from ..tete import TeteTranspose

            self.tete = TeteTranspose(solver)  # raises with MPI
            self.solver = solver
            self.xp = solver.E.xp
            self.N = solver.N
            self.dt = solver.dt
            self.cpml = solver.activate_cpml
            self.dtype = solver.E.array.dtype
            self.lam_star = None
            self.fused = False
            self.lumped, self.dispersive, self.tensor, self.beams = [], [], [], []
            self.dispersive_h = []
            self.psi_names = self.tete.psi_names
            return
        if self.mpi and (
            solver.activate_cpml
            or lumped_elements
            or dispersive
            or dispersive_h
            or tensor
            or beam_objs
        ):
            raise NotImplementedError(
                "With MPI, the adjoint supports the standard step only (no CPML, "
                "lumped elements, dispersive media, tensor averaging or beams)."
            )

        self.solver = solver
        self.xp = xp = solver.E.xp
        self.N = solver.N
        self.dt = solver.dt
        self.cpml = solver.activate_cpml
        self.dtype = solver.E.array.dtype
        self.ieps = xp.array(solver.ieps.toarray())
        self.imu = xp.asarray(solver.imu.toarray())
        self.loss = None
        if solver.use_conductivity:
            self.loss = 1.0 - self.dt * xp.asarray(solver.sigma.toarray()) * self.ieps
        self.lam_star = None

        self.lumped = [
            _LumpedTranspose(el, f"JL{i}", self.dt, self.ieps)
            for i, el in enumerate(lumped_elements)
        ]
        self.dispersive = [
            _DispersiveTranspose(g, gi)
            for gi, g in enumerate(getattr(solver, "_dispersive_groups", []))
        ]
        if network is not None:  # dispersive subpixel averaging
            from ..tensor_dispersive import NetworkTranspose

            self.dispersive.append(NetworkTranspose(network))
        self.dispersive_h = [
            _DispersiveTranspose(g, gi, field="H", prefix="m", s_key="mu_s")
            for gi, g in enumerate(dispersive_h)
        ]
        self.tensor = [] if tensor is None else [_TensorTranspose(tensor, solver)]
        self.beams = [
            BeamTranspose(b, f"beam{i}", self.dt, self.ieps)
            for i, b in enumerate(beam_objs)
        ]
        corrected = [
            ext.idx_host for ext in self.lumped + self.dispersive + self.tensor
        ]
        if corrected:
            joined = np.concatenate(corrected)
            if np.unique(joined).size != joined.size:
                raise NotImplementedError(
                    "Lumped elements must not share edges with each other, "
                    "with dispersive media or with tensor-averaged edges in the "
                    "adjoint."
                )
        corrected_h = [ext.idx_host for ext in self.dispersive_h]
        if corrected_h:
            joined_h = np.concatenate(corrected_h)
            if np.unique(joined_h).size != joined_h.size:
                raise NotImplementedError(
                    "Magnetic dispersive groups must not share edges in the adjoint."
                )

        if not self.cpml:
            # Hermitian transposes conjugate the Bloch phases of the curls
            self.At = _hermitian(solver.tDsiDmuiDaC, solver)
            self.Bt = _hermitian(solver.itDaiDepsDstC, solver)
            self.psi_names = []
            self._init_fused()
            return

        self.ops_t = {
            term[4]: _hermitian(getattr(solver, term[4]), solver)
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
                            xp.asarray(getattr(solver, f"pml_b_{field}_{d}_{side}")),
                            xp.asarray(getattr(solver, f"pml_c_{field}_{d}_{side}")),
                            xp.asarray(getattr(solver, f"idx_{d}_{side}")),
                        )
                    )
                self.faces[op] = faces
        self.psi_names = [f[0] for faces in self.faces.values() for f in faces]
        self._init_fused()

    def _init_fused(self):
        """
        Fused transposed step for solvers stepping with `wakis.gpu_fused`.

        The transposed step then runs in four kernels (two per half step:
        psi and multiplier update, then the merged transposed curl rows of
        the forward kernels). Its outputs (E, H and psi multipliers) live in
        two buffers used alternately: arrays returned by `step` are
        overwritten two calls later (copy them to keep them).
        """
        from ..gpu_fused import is_fused, tname

        self.fused = is_fused(self.solver) and not self.mpi and not self.dispersive_h
        if not self.fused:
            return
        xp, s = self.xp, self.solver
        fs = s._fused_step
        if not fs.ready:
            fs._build()
        self._T = tname(self.dtype)
        n = 3 * self.N
        if self.cpml:
            self._halves = {"E": fs.e.transpose(), "H": fs.h.transpose()}
            if sorted(fs.e.names + fs.h.names) != sorted(self.psi_names):
                raise RuntimeError("Fused and unfused CPML faces differ.")
        else:
            self._ones = xp.ones(n)
        self._beta = xp.zeros(2 * n, dtype=self.dtype)
        self._bufs = []
        for _ in range(2):
            buf = {"E": xp.zeros(n, self.dtype), "H": xp.zeros(n, self.dtype)}
            views = {}
            if self.cpml:
                for key, half in self._halves.items():
                    buf["psi" + key] = xp.zeros(max(half.n_psi, 1), self.dtype)
                    o = half.offsets
                    for k, name in enumerate(half.names):
                        views[name] = buf["psi" + key][o[k] : o[k + 1]]
            buf["views"] = views
            self._bufs.append(buf)

    def _fused_psi_in(self, lam, out_buf):
        """Psi multiplier buffers holding lam's psi arrays (copied if needed)."""
        for buf in self._bufs:
            if buf is not out_buf and all(
                lam[name] is v for name, v in buf["views"].items()
            ):
                return buf
        scratch = self._bufs[0] if out_buf is self._bufs[1] else self._bufs[1]
        for name, v in scratch["views"].items():
            v[...] = lam[name]
        return scratch

    def _fused_base_step(self, lam, lam_E):
        """`_base_step` with the fused kernels (see `_init_fused`)."""
        from ..gpu_fused import launch, tname

        n = 3 * self.N
        T = self._T
        dt = np.float64(self.dt)
        out = self._bufs[1] if lam["H"] is self._bufs[0]["H"] else self._bufs[0]
        if out["E"] is lam_E or out["E"] is lam["E"]:
            raise RuntimeError("Adjoint state aliases the output buffer.")
        if not self.cpml:
            beta = self._beta
            launch(f"adj_beta<{T}>", n, np.int32(n), lam_E, self._ones, dt,
                   None, None, None, None, None, None, beta)  # fmt: skip
            Bt = self.Bt
            launch(f"adj_spmv<{T},{tname(Bt.data.dtype)}>", n, np.int32(n),
                   Bt.indptr, Bt.indices, Bt.data, beta, lam["H"], None,
                   out["H"])  # fmt: skip
            launch(f"adj_beta<{T}>", n, np.int32(n), out["H"], self._ones, -dt,
                   None, None, None, None, None, None, beta)  # fmt: skip
            At = self.At
            launch(f"adj_spmv<{T},{tname(At.data.dtype)}>", n, np.int32(n),
                   At.indptr, At.indices, At.data, beta, lam_E, self.loss,
                   out["E"])  # fmt: skip
            return {"H": out["H"], "E": out["E"]}
        src = self._fused_psi_in(lam, out)
        he, hh = self._halves["E"], self._halves["H"]
        beta = self._beta
        # Reverse of the E half step
        launch(f"adj_beta<{T}>", n, np.int32(n), lam_E, self.ieps, dt, he.slot_a,
               he.slot_b, src["psiE"], out["psiE"], he.pb, he.pc, beta)  # fmt: skip
        launch(f"adj_spmv<{T},{tname(he.t_vdtype)}>", n, np.int32(n), he.t_ptr,
               he.t_col, he.t_val, beta, lam["H"], None, out["H"])  # fmt: skip
        # Reverse of the H half step (kappa = -dt imu lam_H)
        launch(f"adj_beta<{T}>", n, np.int32(n), out["H"], self.imu, -dt,
               hh.slot_a, hh.slot_b, src["psiH"], out["psiH"], hh.pb, hh.pc,
               beta)  # fmt: skip
        launch(f"adj_spmv<{T},{tname(hh.t_vdtype)}>", n, np.int32(n), hh.t_ptr,
               hh.t_col, hh.t_val, beta, lam_E, self.loss, out["E"])  # fmt: skip
        res = {"H": out["H"], "E": out["E"]}
        res.update(out["views"])
        return res

    def _comp(self, array, c):
        i = "xyz".index(c)
        return array[i * self.N : (i + 1) * self.N]

    def zero_state(self):
        """Zero adjoint state: dict with 'H', 'E', CPML psi and auxiliary arrays."""
        state = {k: self.xp.zeros_like(v) for k, v in self.get_state().items()}
        state["H"] = self.xp.zeros(3 * self.N, dtype=self.dtype)
        state["E"] = self.xp.zeros(3 * self.N, dtype=self.dtype)
        return state

    def get_state(self):
        """Copy of the solver's forward state (fields, psi, auxiliary arrays)."""
        state = {
            "H": self.solver.H.toarray().copy(),
            "E": self.solver.E.toarray().copy(),
        }
        for name in self.psi_names:
            state[name] = getattr(self.solver, name).copy()
        for ext in self.lumped + self.dispersive + self.dispersive_h + self.beams:
            ext.get_state(state)
        return state

    def set_state(self, state):
        """
        Overwrite the solver's forward state with the arrays in `state`.

        The impressed and conduction currents (solver.J, solver.M and their
        bookkeeping) are reset to zero, so that the next forward step applies
        the homogeneous step U to `state`. With MPI, the ghost layers of E and
        H are then filled from the neighbouring ranks, as after a forward step.
        """
        solver, xp = self.solver, self.xp
        solver.H.fromarray(xp.asarray(state["H"]))
        solver.E.fromarray(xp.asarray(state["E"]))
        solver.J.fromarray(xp.zeros(3 * self.N, dtype=self.dtype))
        if solver.M is not None:
            solver.M.fromarray(xp.zeros(3 * self.N, dtype=self.dtype))
        if hasattr(solver, "J_old"):
            solver.J_old = xp.zeros(3 * self.N, dtype=self.dtype)
        for name in self.psi_names:
            if getattr(solver, "fused", False):
                # In place: the fused step owns the psi buffers
                getattr(solver, name)[...] = xp.asarray(state[name])
            else:
                setattr(solver, name, xp.array(state[name]))
        for ext in self.lumped + self.dispersive + self.dispersive_h + self.beams:
            ext.set_state(state, xp)
        if self.mpi:
            solver._mpi_communicate(solver.E)
            solver._mpi_communicate(solver.H)

    def inner(self, x, y):
        """
        Inner product sum_k <x_k, y_k> of two states, conjugating x.

        With MPI, the ghost layers are left out and the result is summed over
        the ranks: it is the inner product of the global states.

        Parameters
        ----------
        x, y : dict of ndarray
            States with the keys of `zero_state`.

        Returns
        -------
        complex
        """
        total = 0.0
        for k in x:
            a, b = x[k], y[k]
            if self.mpi and k in ("E", "H"):
                mask = self._owned_mask()
                a, b = a[mask], b[mask]
            total += complex(self.xp.vdot(a, b))
        if self.mpi:
            total = self.solver.grid.comm.allreduce(total)
        return total

    def _ghost_layers(self):
        """Flat slices of the low and high MPI ghost z-layers of E and H."""
        grid = self.solver.grid
        nxy, nz = self.solver.Nx * self.solver.Ny, self.solver.Nz
        low, high = [], []
        for c in range(3):
            base = c * self.N
            if grid.rank > 0:
                low.append(slice(base, base + nxy))
            if grid.rank < grid.size - 1:
                high.append(slice(base + (nz - 1) * nxy, base + nz * nxy))
        return low, high

    def _owned_mask(self):
        mask = self.xp.ones(3 * self.N, dtype=bool)
        low, high = self._ghost_layers()
        for sl in low + high:
            mask[sl] = False
        return mask

    def _fold_ghosts(self, arrays):
        """
        Transpose of the MPI ghost exchange, applied in place.

        The forward exchange copies the first (last) owned z-layer of each
        rank into the high (low) ghost layer of the rank below (above) and
        discards the ghost values computed locally. Its transpose adds the
        adjoint held on each ghost layer to the owned layer it was copied
        from, and zeroes the ghost layer.
        """
        grid, xp = self.solver.grid, self.xp
        comm, rank = grid.comm, grid.rank
        nxy = self.solver.Nx * self.solver.Ny
        low, high = self._ghost_layers()

        def exchange(ghosts, neighbour, shift, tags):
            send = _host(xp.stack([a[sl] for a in arrays for sl in ghosts]))
            send = np.ascontiguousarray(send)
            recv = np.empty_like(send)
            comm.Sendrecv(send, dest=neighbour, sendtag=tags[0], recvbuf=recv,
                          source=neighbour, recvtag=tags[1])  # fmt: skip
            recv = xp.asarray(recv)
            i = 0
            for a in arrays:
                for sl in ghosts:
                    # The neighbour's ghost next to ours is our owned layer
                    a[sl.start + shift : sl.stop + shift] += recv[i]
                    i += 1

        # Same pairing order as SolverFIT3D._mpi_communicate
        if low:
            exchange(low, rank - 1, nxy, (2, 3))
        if high:
            exchange(high, rank + 1, -nxy, (3, 2))
        for a in arrays:
            for sl in low + high:
                a[sl] = 0

    def step(self, lam):
        """
        Apply the transposed time step to adjoint variables.

        Parameters
        ----------
        lam : dict of ndarray
            Adjoint variables of the state after a step (keys as in
            `zero_state`, arrays on the solver's device). With MPI, values on
            ghost layers (e.g. adjoint sources of monitors reading ghost
            samples) are first added to the rank that owns the sample.

        Returns
        -------
        dict of ndarray
            Adjoint variables of the state before that step (zero on MPI
            ghost layers). The adjoint variable of the E computed by the
            solver step itself (before lumped and dispersive corrections) is
            kept in `lam_star`.
        """
        if self.mpi:
            lam = dict(lam, E=lam["E"].copy(), H=lam["H"].copy())
            self._fold_ghosts([lam["E"], lam["H"]])
        corrections = self.lumped + self.dispersive + self.tensor
        out = {}
        if not corrections and not self.beams:
            # The solver step is the whole step: lam_E* = lam_E'
            lam_star = lam["E"]
            out.update(self._base_step(lam, lam_star))
            self.lam_star = lam_star
            return out
        lam_star = lam["E"].copy() if corrections else lam["E"]
        extra = self.xp.zeros_like(lam_star) if corrections else None
        for ext in corrections:
            ext.transpose_post(lam, lam_star, extra, out)
        out.update(self._base_step(lam, lam_star))
        for ext in self.lumped + self.beams:
            ext.transpose_pre(lam, lam_star, out)
        if extra is not None:
            out["E"] = out["E"] + extra
        self.lam_star = lam_star
        return out

    def _transpose_dispersive_h(self, lam, lam_H, out):
        """Transpose of `_update_dispersive_H`: returns (lam_H_star, extra_H)."""
        if not self.dispersive_h:
            return lam_H, None
        lam_H_star = lam_H.copy()
        extra_H = self.xp.zeros_like(lam_H)
        lam_h_in = dict(lam, H=lam_H)
        for ext in self.dispersive_h:
            ext.transpose_post(lam_h_in, lam_H_star, extra_H, out)
        return lam_H_star, extra_H

    def _base_step(self, lam, lam_E):
        """Transpose of the solver step, given the adjoint lam_E of its E."""
        if self.tete is not None:
            return self.tete.step(dict(lam, E=lam_E))
        if self.fused:
            return self._fused_base_step(lam, lam_E)
        out = {}
        if not self.cpml:
            # In place: lam_H = lam_H' + dt B^H lam_E, lam_E = (1 - S) lam_E - dt A^H lam_H*
            lam_H = self.Bt @ lam_E
            lam_H *= self.dt
            lam_H += lam["H"]
            if self.mpi:
                # Reverse of the H exchange between the two half steps
                self._fold_ghosts([lam_H])
            lam_H_star, extra_H = self._transpose_dispersive_h(lam, lam_H, out)
            curl = self.At @ lam_H_star
            curl *= self.dt
            lam_E = lam_E.copy() if self.loss is None else self.loss * lam_E
            lam_E -= curl
            if self.mpi:
                # Reverse of the E exchange that ends the forward step
                self._fold_ghosts([lam_E])
            out["H"] = lam_H if extra_H is None else lam_H_star + extra_H
            out["E"] = lam_E
            return out

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

        lam_H_star, extra_H = self._transpose_dispersive_h(lam, lam_H, out)

        # Reverse of the H half step: H* = H - dt imu s (D E + psiH')
        kappa = -self.dt * self.imu * lam_H_star
        for T, d, S, s, op, prefix in _H_TERMS:
            kappa_T = self._comp(kappa, T)
            alpha = s * kappa_T
            for name, b, c, idx in self.faces[op]:
                mu = lam[name] + s * kappa_T[idx]
                alpha[idx] += c * mu
                out[name] = b * mu
            self._comp(lam_E, S)[:] += self.ops_t[op] @ alpha
        out["H"] = lam_H if extra_H is None else lam_H_star + extra_H
        out["E"] = lam_E
        return out


class AdjointSimulation:
    def __init__(
        self,
        make_solver,
        sources,
        monitors,
        design,
        Nt,
        store_dtype=np.float64,
        store_on_host=None,
        memory="store",
        checkpoints=None,
        block=None,
        memory_budget_gb=None,
        decimation=None,
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

        Both runs use the solver's backend: CuPy arrays on the GPU with
        ``use_gpu``, and the z-slab decomposition with ``use_mpi`` (run the
        script under ``mpiexec``; every rank then returns the same global
        outputs and gradient).

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
            `readouts.outputs` for the raw outputs of each. With MPI, only
            `DFTMonitor` and `FluxMonitor`.
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
        store_on_host : bool, optional
            GPU only: keep the stored increments in host memory (one row is
            copied per time step). By default this is done when they would
            take more than half of the free GPU memory.
        memory : {'store', 'checkpoint', 'reverse', 'auto', 'nyquist'}
            How the backward sweep obtains the per-step design rows (see
            `wakis.inverse.memory`). 'store' (default): keep all Nt rows.
            'checkpoint': binomial checkpointing with replay, bitwise equal
            gradients for all physics. 'reverse': time reversal of the
            lossless leapfrog with recorded CPML interface layers (falls
            back to 'checkpoint' with a warning when the problem has loss,
            dispersion, lumped elements or tensor averaging); agrees with
            'store' to round-off. 'auto': 'store' if the rows fit in
            `memory_budget_gb`, else 'checkpoint' with slots and block
            chosen by `plan_checkpoints`. 'nyquist': APPROXIMATE
            decimated storage (requires `decimation`); never exact.
        checkpoints : int, optional
            'checkpoint': number of solver snapshots held at once (>= 1).
        block : int, optional
            'checkpoint': steps per block (rows of one block are held during
            the backward sweep). With `checkpoints` and without `block`,
            the steps are split into binom(checkpoints + 2, 2) blocks, so
            that each step is recomputed at most twice.
        memory_budget_gb : float, optional
            'auto' and 'checkpoint' (without `checkpoints`): budget for the
            snapshots and rows [GB]. Default: half of the free GPU memory
            (GPU) or 25 % of the available host memory.
        decimation : int, optional
            'nyquist': keep every `decimation`-th row.
        """
        for name, monitor in monitors.items():
            readouts.check(name, monitor)
        self.make_solver = make_solver
        self.sources = list(sources)
        self.monitors = dict(monitors)
        self.design = design
        self.Nt = int(Nt)
        self.store_dtype = store_dtype
        self.store_on_host = store_on_host
        if memory not in ("store", "checkpoint", "reverse", "auto", "nyquist"):
            raise ValueError(f"Unknown memory mode {memory!r}.")
        if memory == "nyquist" and not decimation:
            raise ValueError("memory='nyquist' needs a decimation factor.")
        self.memory = memory
        self.checkpoints = checkpoints
        self.block = block
        self.memory_budget_gb = memory_budget_gb
        self.decimation = decimation
        self.memory_stats = None
        self.rho = None
        self.readouts = None
        # Fused GPU solvers: replay the time loops as CUDA graphs when every
        # per-step operation is device resident (see wakis.gpu_fused)
        self.use_graphs = True

    def _comm(self, solver):
        """MPI communicator of a decomposed solver, or None."""
        if not (solver.use_mpi and solver.grid.use_mpi):
            return None
        for name, monitor in self.monitors.items():
            if type(monitor) not in (readouts.DFTMonitor, readouts.FluxMonitor):
                raise NotImplementedError(
                    f"Monitor {name!r}: with MPI the adjoint supports DFTMonitor "
                    "and FluxMonitor only."
                )
        return solver.grid.comm

    def output_shapes(self):
        """
        Shapes of the arrays returned by `forward`, without time-stepping.

        Also stores in `readouts` copies of the monitors initialized on a
        fresh solver (sample geometry and weights), for the helper functions
        that turn raw outputs into flux, mode amplitudes or far fields. With
        MPI, the flux weights of all ranks are gathered in the order of the
        gathered outputs.

        Returns
        -------
        dict
            {monitor name: {output key: shape}}.
        """
        solver = self.make_solver()
        if getattr(solver, "scheme", "yee") == "tete":
            # ieps enters each TE/TM step in three sub-updates (two half kicks
            # of ex, ey): the single-increment gradient formula does not apply
            raise NotImplementedError(
                "AdjointSimulation (design gradients) does not support "
                "scheme='tete'; AdjointStepper does."
            )
        comm = self._comm(solver)
        self.readouts = {}
        shapes = {}
        for name, template in self.monitors.items():
            monitor = copy.deepcopy(template)
            readouts.initialize(monitor, solver)
            self.readouts[name] = monitor
            shapes[name] = readouts.shapes(monitor)
            if comm is None:
                continue
            for key, shape in shapes[name].items():
                total = sum(comm.allgather(shape[-1]))
                shapes[name][key] = (*shape[:-1], total)
            for term in getattr(monitor, "_terms", []):
                term["weight"] = np.concatenate(comm.allgather(term["weight"]))
        return shapes

    def _store_on_host(self, solver, nbytes):
        if not solver.use_gpu:
            return False
        if self.store_on_host is not None:
            return bool(self.store_on_host)
        free, _ = solver.E.xp.cuda.Device().mem_info
        return nbytes > free / 2

    def forward(self, rho, record=True):
        """
        Run the simulation for density rho.

        With ``record=False`` nothing is kept for a gradient (no design rows,
        snapshots or interface records): use it for objective-only
        evaluations; `vjp` then raises until a recording forward run.

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
            [field units x s]. With MPI, the samples of all ranks are
            concatenated along the last axis in rank order (z for a
            `DFTMonitor`), identically on every rank.
        """
        rho = np.array(rho, dtype=float)
        solver = self.make_solver()
        comm = self._comm(solver)
        self.design.apply(solver, rho)
        sources, monitors = copy.deepcopy((self.sources, self.monitors))
        elements = [src for src in sources if hasattr(src, "post_step")]
        for name, monitor in monitors.items():
            if isinstance(monitor, readouts.LumpedReadout) and not any(
                monitor.element is el for el in elements
            ):
                raise ValueError(f"The element of readout {name!r} is not a source.")

        xp = solver.E.xp
        disp = self.design.dispersive_group(solver)
        net = self.design.network_of(solver)
        if net is not None:
            return self._forward_network(
                rho, solver, net, sources, monitors, elements, comm, record
            )
        self._tensor = self._tensor_tracking(solver, disp)
        idx = xp.asarray(
            self.design.flat_idx if self._tensor is None else self._tensor["track"]
        )
        complex_fields = np.iscomplexobj(solver.E.array)
        store = np.result_type(self.store_dtype, np.complex64 if complex_fields else 1)
        ieps_d = solver.ieps.array[idx].copy()
        tensor = self._tensor

        def advance(n, record, monitors=(), mid=None):
            t = n * solver.dt
            for src in sources:
                src.update(solver, t)
            if mid is not None:
                mid(n)
            row = None
            if not record:
                solver.one_step()
            elif disp is None:
                e_old = solver.E.array[idx]
                solver.one_step()
                delta = solver.E.array[idx] - e_old
                if tensor is not None:
                    # Increment of the diagonal step (before the correction)
                    delta[tensor["rows"]] = solver.tensor_operator.last_increment
                row = delta / ieps_d
            else:
                # The ADE overwrites the design edges: record the parameter
                # sensitivity of the correction instead (see MaterialGrid)
                e_old = solver.E.array[idx]
                state_old = self.design.ade_state(disp, solver)
                solver.one_step()
                row = self.design.ade_sensitivity(disp, solver, e_old, state_old)
            for el in elements:
                el.post_step(solver, t)
            for monitor in monitors:
                monitor.update(solver, t)
            return row

        if record:
            rung = self._make_rung(
                solver, sources, advance, (idx.size,), store, idx, ieps_d, disp
            )
        else:
            rung = mem.NullRung()
        fused_plain = getattr(solver, "fused", False) and disp is None and (
            self._tensor is None
        )
        if fused_plain and not record:
            # Objective only: the CUDA-graph loop without any recording
            self._fused_forward(solver, sources, monitors, elements, idx, None)
            rung.finish()
            self.memory_stats = rung.stats()
        elif self._fused_store(solver, rung, disp):
            # Fused GPU solver, all rows stored on the device: record them in
            # the CUDA-graph time loop (wakis.gpu_fused)
            self._fused_forward(solver, sources, monitors, elements, idx, rung.rows)
            rung.finish()
            self.memory_stats = rung.stats()
        else:
            self._run_forward(rung, advance, monitors)
        increments = rung.rows if isinstance(rung, mem.StoreRung) else None
        self._check_overlaps(solver, elements, monitors)
        self._solver, self._monitors, self._elements = solver, monitors, elements
        self._increments = increments
        self._rung = rung
        self._recorded = bool(record)
        self.rho = rho
        raw = {name: readouts.outputs(m) for name, m in monitors.items()}
        if comm is not None:
            raw = self._gather(raw, comm)
        return raw

    def _fused_forward(self, solver, sources, monitors, elements, idx, increments):
        """
        Time loop of `forward` on a fused GPU solver: the design increments
        q^n / ieps = (E*^{n+1} - E^n) / ieps are recorded by two kernels per
        step (gather of E^n, then the scaled difference into its row).
        """
        from ..gpu_fused import FusedLoop, idx32, launch, rname, tname

        rec = increments is not None  # False: objective-only run
        idx_d = idx32(idx)
        ieps_d = solver.ieps.array[idx_d].astype(np.float64)
        e_old = solver.E.xp.empty(idx_d.size if rec else 0, dtype=solver.E.array.dtype)
        T = tname(solver.E.array.dtype)
        R = rname(increments.dtype) if rec else None
        m = np.int32(idx_d.size)
        mons = list(monitors.values())
        graph = self.use_graphs and FusedLoop.supported(solver, sources, mons, elements)
        for n in range(self.Nt):
            if graph and n == 1:
                # Steps 1 .. Nt-1 as CUDA graph replays (step 0 initialized all)
                loop = FusedLoop(solver, sources, mons, self.Nt,
                                 record=(idx_d, ieps_d, e_old, increments)
                                 if rec else None)  # fmt: skip
                loop.run(1, self.Nt)
                loop.close()
                break
            t = n * solver.dt
            for src in sources:
                src.update(solver, t)
            if rec:
                launch(f"gather<{T}>", m, m, idx_d, solver.E.array, e_old)
            solver.one_step()
            if rec:
                launch(f"record_row<{T},{R}>", m, m, idx_d, solver.E.array, e_old,
                       ieps_d, increments, None, np.int64(0), np.int64(n))  # fmt: skip
            for el in elements:
                el.post_step(solver, t)
            for monitor in mons:
                monitor.update(solver, t)

    def _budget_bytes(self, solver):
        if self.memory_budget_gb is not None:
            return self.memory_budget_gb * 1e9
        if solver.use_gpu:
            free, _ = solver.E.xp.cuda.Device().mem_info
            return free / 2
        return 0.25 * mem.available_host_bytes()

    def _make_rung(self, solver, sources, advance, row_shape, dtype, idx, ieps_d,
                   disp):  # fmt: skip
        """Memory rung of this forward run (see `wakis.inverse.memory`)."""
        from . import memory as mem

        xp = solver.E.xp
        mode = self.memory
        row_bytes = int(np.prod(row_shape)) * np.dtype(dtype).itemsize
        # Rows held by the checkpoint rung (snapshot replays) keep the full
        # precision of the step (float64 / complex128), whatever store_dtype
        exact = np.result_type(solver.E.array.dtype, np.float64)
        exact_row_bytes = int(np.prod(row_shape)) * np.dtype(exact).itemsize
        if mode == "reverse":
            reason = None
            if idx is None:
                reason = "a dispersive tensor design"
            else:
                reason = mem.reverse_supported(solver, sources, self.design)
            if reason is None:
                return mem.ReverseRung(self.Nt, solver, sources, idx, ieps_d, dtype)
            mem.warn_fallback(reason)
            mode = "checkpoint"
        if mode == "nyquist":
            return mem.NyquistRung(self.Nt, self.decimation, row_shape, dtype, xp)
        io = mem.SolverState(solver, sources)
        slots, block = self.checkpoints, self.block
        if mode == "auto" or (mode == "checkpoint" and slots is None):
            # Snapshot size: fields, CPML psi and auxiliary arrays (sources
            # are small); measured exactly when the first snapshot is taken
            state_bytes = 2 * solver.E.array.nbytes + solver.J.array.nbytes
            state_bytes += sum(
                getattr(solver, k).nbytes for k in io.psi
            ) + (0 if solver.M is None else solver.M.array.nbytes)
            budget = self._budget_bytes(solver)
            if mode == "checkpoint":
                # Explicit checkpointing: at most a quarter of the stored rows
                budget = min(budget, 0.25 * self.Nt * exact_row_bytes)
            try:
                if mode == "auto" and self.Nt * row_bytes <= budget:
                    plan = dict(mode="store")  # stored rows (store_dtype) fit
                else:
                    plan = mem.plan_checkpoints(
                        self.Nt, state_bytes, exact_row_bytes, budget
                    )
            except MemoryError:
                if mode == "auto":
                    raise
                plan = dict(mode="checkpoint", slots=2, block=None)
            if mode == "auto" and plan["mode"] == "store":
                mode = "store"
            else:
                mode = "checkpoint"
                slots = slots or plan["slots"]
                block = block or plan["block"]
        if mode == "store":
            on_host = self._store_on_host(solver, self.Nt * row_bytes)
            return mem.StoreRung(self.Nt, row_shape, dtype, xp, on_host)
        if block is None:
            n_blocks = mem.beta(slots, 2)
            block = max(1, -(-(self.Nt - 1) // n_blocks))
            # ... but never more rows than the memory budget holds
            state = 2 * solver.E.array.nbytes + solver.J.array.nbytes
            fit = (self._budget_bytes(solver) - slots * state) // exact_row_bytes
            block = int(max(1, min(block, fit)))
        return mem.CheckpointRung(self.Nt, slots, block, advance, io, dtype, xp,
                                  verify=getattr(self, "verify_replay", False))  # fmt: skip

    def _fused_store(self, solver, rung, disp):
        """Whether the fused CUDA-graph loops apply: fused solver, no ADE or
        tensor design, and the effective memory mode is 'store' on the device
        (explicit, or 'auto' resolved to it). Other rungs use the generic
        loop, whose `solver.one_step` is the fused step."""
        return (
            getattr(solver, "fused", False)
            and disp is None
            and self._tensor is None
            and isinstance(rung, mem.StoreRung)
            and not rung.on_host
        )

    def _run_forward(self, rung, advance, monitors):
        """The forward time loop, feeding the memory rung."""
        monitors = list(monitors.values())
        mid = rung.mid_step
        for n in range(self.Nt):
            rung.before_step(n)
            row = advance(n, rung.wants_row(n), monitors, mid)
            rung.after_step(n, row)
        rung.finish()
        self.memory_stats = rung.stats()

    def _forward_network(self, rho, solver, net, sources, monitors, elements, comm,
                         record_rows=True):  # fmt: skip
        """
        Time loop for a dispersive tensor design (`MaterialGrid` with
        `material` and ``smoothing='tensor'``): records, per step, the terms
        of the sub-capacitor update that depend on the design parameters
        (dQ, u^n, u^{n+1} and the pole sums, see `_vjp_network`).
        """
        if comm is not None:
            raise NotImplementedError("Dispersive tensor designs do not support MPI.")
        net.record = True
        xp = solver.E.xp

        def advance(n, record, monitors=(), mid=None):
            t = n * solver.dt
            for src in sources:
                src.update(solver, t)
            net.record = bool(record)
            solver.one_step()
            row = xp.stack(net.last) if record else None
            for el in elements:
                el.post_step(solver, t)
            for monitor in monitors:
                monitor.update(solver, t)
            return row

        if record_rows:
            rung = self._make_rung(
                solver, sources, advance, (5, net.n_caps), np.float64, None, None, None
            )
        else:
            rung = mem.NullRung()
        self._run_forward(rung, advance, monitors)
        self._recorded = bool(record_rows)
        self._rung = rung
        net.record = False
        self._check_overlaps(solver, elements, monitors)
        self._solver, self._monitors, self._elements = solver, monitors, elements
        self._increments = getattr(rung, "rows", None)
        self._tensor = None
        self.rho = rho
        return {name: readouts.outputs(m) for name, m in monitors.items()}

    def _vjp_network(self, stepper, taps, net):
        """
        Gradient of a dispersive tensor design. With the stored dQ, u, u', the
        pole sums L = sum_L (p'_L - p_L) and Dsum = sum_D (kappa - 1) p_D of
        each capacitor, lam_E the adjoint of E^{n+1} on its edge and lam_u the
        total adjoint of u' (`NetworkTranspose.lam_unew`)::

            dF/dieps  = sum_n lam_E eps_0 dQ
            dF/dc     = sum_n lam_E (u' - u - dQ / eps_inf)
            dF/deps_inf = sum_n [lam_u (u - u') / den + lam_E c dQ / eps_inf^2]
            dF/dmu    = sum_n lam_u (-Ls u - L - Dsum - Ls u') / den

        with Ls = sum_D beta + sigma dt / (2 eps_0) of the material.
        """
        solver = self._solver
        xp = stepper.xp
        dt = solver.dt
        ext = next(e for e in stepper.dispersive if getattr(e, "net", None) is net)
        rows = net.rows_dev
        cap_row = xp.asarray(net.cap_row)
        c = xp.asarray(net.c_host.sum(axis=0).A1)  # one edge per capacitor
        eps_inf, den = net.eps_inf, net.den
        mat = net.materials[0]
        _, debye = _ade_coefficients(mat, dt)
        Ls = sum(b for _, b in debye) + mat.sigma * dt / eps_0 / 2
        g_ieps = xp.zeros(net.n_caps)
        g_c = xp.zeros(net.n_caps)
        g_eps = xp.zeros(net.n_caps)
        g_mu = xp.zeros(net.n_caps)
        lam = stepper.zero_state()
        get_row = self._rung.reader()
        for n in reversed(range(self.Nt)):
            t = n * dt
            for tap in taps:
                if tap.skip_last and n == self.Nt - 1:
                    continue
                w_conj = dt * xp.exp(2j * np.pi * tap.freqs * (t + tap.offset))
                source = tap.WH @ (w_conj @ tap.G)
                lam[tap.field][tap.rows] += source.real
            lam_E = lam["E"][rows][cap_row]
            lam = stepper.step(lam)
            lam_u = ext.lam_unew
            dQ, u, u1, lsum, dsum = (xp.asarray(a) for a in get_row(n))
            g_ieps += lam_E * eps_0 * dQ
            g_c += lam_E * (u1 - u - dQ / eps_inf)
            g_eps += lam_u * (u - u1) / den + lam_E * c * dQ / eps_inf**2
            g_mu += lam_u * (-Ls * u - lsum - dsum - Ls * u1) / den
        n_e = self.design.flat_idx.size
        # ieps is per edge: both capacitors of an edge carry the same dQ and lam_E
        g_ieps = _host(g_ieps)[:n_e]
        return self.design._network_vjp(
            self.rho, g_ieps, _host(g_c), _host(g_eps), _host(g_mu)
        )

    def _fused_vjp(self, stepper, taps, idx):
        """
        Backward loop of `vjp` with fused kernels: each adjoint DFT source
        costs two kernels (phasor-weighted cotangent sum, then the W^H rows),
        the transposed step four and the gradient accumulation one.
        """
        from ..gpu_fused import idx32, launch, rname, tname

        xp = stepper.xp
        dt = self._solver.dt
        idx_d = idx32(idx)
        m = np.int32(idx_d.size)
        inc = self._increments
        T, R = tname(stepper.dtype), rname(inc.dtype)
        dev = []
        for tap in taps:
            WH = tap.WH
            dev.append(
                (
                    tap,
                    idx32(tap.rows),
                    xp.ascontiguousarray(WH.data, dtype=np.complex128),
                    xp.ascontiguousarray(tap.G, dtype=np.complex128),
                    xp.empty(WH.shape[1], dtype=np.complex128),
                    xp.asarray(tap.freqs, dtype=np.float64),
                    WH,
                )
            )
        c1_im = np.float64((2j * np.pi).imag)
        grad = xp.zeros(idx_d.size)
        holder = {"lam": stepper.zero_state()}
        counter = xp.zeros(1, dtype=np.int64)

        def one(n=None, last=False):
            # Step n on the host (n given) or at the device counter (graph)
            lam = holder["lam"]
            c = None if n is not None else counter
            t = np.float64(0.0 if n is None else n * dt)
            row = np.int64(0 if n is None else n)
            # A DFT sample X = sum_n w_n W x^{n+1} gives lam += W^H conj(w_n) G
            for tap, rows, val, G, v, freqs, WH in dev:
                if tap.skip_last and last:
                    continue
                k = np.int32(v.size)
                launch("tap_v", k, k, np.int32(freqs.size), freqs, G, c1_im,
                       np.float64(dt), np.float64(tap.offset), t, c, np.int64(0),
                       v)  # fmt: skip
                launch(f"tap_add<{tname(lam[tap.field].dtype)}>", rows.size,
                       np.int32(rows.size), rows, WH.indptr, WH.indices, val, v,
                       lam[tap.field])  # fmt: skip
            lam = stepper.step(lam)
            launch(f"grad_acc<{T},{R}>", m, m, idx_d, stepper.lam_star, inc, c,
                   np.int64(0), row, grad)  # fmt: skip
            if c is not None:
                launch("step_add", 1, c, np.int64(-1))
            holder["lam"] = lam

        corrections = (
            stepper.lumped + stepper.dispersive + stepper.tensor + stepper.beams
        )
        graph = self.use_graphs and not corrections and self.Nt > 1
        if not graph:
            for n in reversed(range(self.Nt)):
                one(n, last=n == self.Nt - 1)
            return grad
        from ..gpu_fused import GraphRunner

        one(self.Nt - 1, last=True)
        counter.fill(self.Nt - 2)
        GraphRunner(one, chunk=64).run(self.Nt - 1)
        return grad

    def _tensor_tracking(self, solver, disp):
        """
        Edges whose step increments the gradient needs with tensor averaging.

        With N = diag(ieps) + K the step is E' = E + N g, g = dt (curl H - J)
        the increment of D, so dF/dieps_a = sum_n lam'_a g_a and
        dF/dS_ab = sum_n (lam'_a g_b / W_a + lam'_b g_a / W_b) for the
        coupling S = W K, with lam' the adjoint of E'. g is recorded on the
        design edges and on all coupled edges (`track`).
        """
        op = getattr(solver, "tensor_operator", None)
        if op is None or op.rows.size == 0:
            return None
        if disp is not None:
            raise NotImplementedError(
                "Dispersive designs are not supported with tensor averaging."
            )
        design = self.design.flat_idx
        track = np.union1d(design, op.rows)
        xp = solver.E.xp
        out = {
            "track": track,
            "rows": xp.asarray(np.searchsorted(track, op.rows)),
            "design": xp.asarray(np.searchsorted(track, design)),
            "pairs": None,
        }
        if self.design.smoothing == "tensor":
            ea, eb = op.pair_a, op.pair_b
            sel = np.isin(ea, design) | np.isin(eb, design)
            ea, eb = ea[sel], eb[sel]
            out["pairs"] = (ea, eb, np.real(op.pair_coef[sel]))
            out["pa"] = xp.asarray(np.searchsorted(track, ea))
            out["pb"] = xp.asarray(np.searchsorted(track, eb))
            out["iWa"] = xp.asarray(1.0 / op.W[ea])
            out["iWb"] = xp.asarray(1.0 / op.W[eb])
        return out

    def _gather(self, raw, comm):
        """
        Concatenate the outputs of all ranks along their last (sample) axis,
        recording the slice of each output held by this rank.
        """
        self._slices = {}
        out = {}
        for name, comps in raw.items():
            out[name] = {}
            for key, local in comps.items():
                sizes = comm.allgather(local.shape[-1])
                start = sum(sizes[: comm.Get_rank()])
                self._slices[name, key] = slice(start, start + local.shape[-1])
                out[name][key] = np.concatenate(comm.allgather(local), axis=-1)
        return out

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
            dF/drho with shape `design.shape` (summed over the MPI ranks).
        """
        if self.rho is None:
            raise RuntimeError("Call forward before vjp.")
        if not getattr(self, "_recorded", True):
            raise RuntimeError("The last forward ran with record=False; rerun it.")
        solver = self._solver
        comm = self._comm(solver)
        stepper = AdjointStepper(solver, self._elements)
        xp = stepper.xp
        dt = solver.dt
        complex_fields = np.iscomplexobj(solver.E.array)

        keys = {id(ext.element): ext.key for ext in stepper.lumped}
        taps = []
        for name, comps in cotangents.items():
            if comm is not None:
                # This rank's samples of the gathered outputs
                comps = {
                    key: np.asarray(G)[..., self._slices[name, key]]
                    for key, G in comps.items()
                }
            taps += readouts.taps(self._monitors[name], solver, comps, keys)
        for tap in taps:
            tap.to_device(xp)

        net = self.design.network_of(solver)
        if net is not None:
            grad = self._vjp_network(stepper, taps, net)
            self.memory_stats = self._rung.stats()
            return grad
        idx = xp.asarray(self.design.flat_idx)
        own = self.design.dispersive_group(solver)
        ade = None
        if own is not None:
            ade = next(ext for ext in stepper.dispersive if ext.group is own)
        tensor = self._tensor
        if (
            stepper.fused
            and tensor is None
            and ade is None
            and isinstance(self._rung, mem.StoreRung)
            and not self._rung.on_host
        ):
            grad = self._fused_vjp(stepper, taps, idx)
            grad = self.design.vjp(self.rho, _host(grad))
            if comm is not None:
                grad = comm.allreduce(grad)
            return grad
        grad = xp.zeros(idx.size)
        if tensor is not None:
            track = xp.asarray(tensor["track"])
            if tensor["pairs"] is not None:
                grad_pairs = xp.zeros(len(tensor["pairs"][0]))
        lam = stepper.zero_state()
        get_row = self._rung.reader()
        for n in reversed(range(self.Nt)):
            t = n * dt
            # A DFT sample X = sum_n w_n W x^{n+1} gives lam += W^H conj(w_n) G
            for tap in taps:
                if tap.skip_last and n == self.Nt - 1:
                    continue
                w_conj = dt * xp.exp(2j * np.pi * tap.freqs * (t + tap.offset))
                source = tap.WH @ (w_conj @ tap.G)
                lam[tap.field][tap.rows] += source if complex_fields else source.real
            if tensor is not None:
                # Adjoint of the corrected E' = E + N g
                lam_new = lam["E"][track]
                g = xp.asarray(get_row(n))
                grad += xp.real(
                    xp.conj(lam_new[tensor["design"]]) * g[tensor["design"]]
                )
                if tensor["pairs"] is not None:
                    pa, pb = tensor["pa"], tensor["pb"]
                    grad_pairs += xp.real(
                        xp.conj(lam_new[pa]) * g[pb] * tensor["iWa"]
                        + xp.conj(lam_new[pb]) * g[pa] * tensor["iWb"]
                    )
                lam = stepper.step(lam)
                continue
            lam = stepper.step(lam)
            sens = stepper.lam_star[idx] if ade is None else ade.lam_acc
            grad += xp.real(xp.conj(sens) * xp.asarray(get_row(n)))
        if tensor is not None and tensor["pairs"] is not None:
            ea, eb, coef = tensor["pairs"]
            grad = self.design.vjp(
                self.rho, _host(grad), (ea, eb, coef * _host(grad_pairs))
            )
        else:
            grad = self.design.vjp(self.rho, _host(grad))
        if comm is not None:
            grad = comm.allreduce(grad)
        self.memory_stats = self._rung.stats()
        return grad
