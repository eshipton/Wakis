# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Exact discrete adjoint and topology-optimization (`MaterialGrid`,
`AdjointSimulation`, `jax_simulation`) adapter for `SubgridFIT2D`.

Every step of `SubgridFIT2D` (including hanging-node prolongation `P`,
sub-pixel material averaging, Berenger split-field PML, and `SubgridDFTMonitor`
/ `SubgridModeMonitor` frequency-domain accumulation) is linear in the state
`(u, f, ux, uy)`. `SubgridAdjointSimulation` implements the exact reverse-mode
transpose (`vjp`) and conforms to the interface of `wakis.inverse.jax_simulation`
so that `jax.grad` and `jax.value_and_grad` work directly on subgrid S-parameter
and field objectives.
"""

import numpy as np
import scipy.sparse as sp
from scipy.constants import epsilon_0 as EPS0

from .monitors import SubgridDFTMonitor, SubgridModeMonitor

__all__ = [
    "SubgridMaterialGrid",
    "SubgridAdjointSimulation",
    "subgrid_mode_amplitudes",
]


class SubgridMaterialGrid:
    """
    2D design region mapping pixel densities `rho` in `[0, 1]` of shape
    `(nx, ny)` to the node masses `D` (in TM) or edge Hodges `W` (in TE) of a
    `SubgridFIT2D` solver.

    Uses the SIMP interpolation

    .. math::

        \\varepsilon(\\rho) = \\varepsilon_{\\min}
        + (\\varepsilon_{\\max} - \\varepsilon_{\\min}) \\rho^p

    and exact analytical rectangle-rectangle overlap areas between the
    `(nx, ny)` design pixels and the leaf-cell node quadrants (TM) or dual-edge
    cells (TE).

    Parameters
    ----------
    center : sequence of float
        Centre `(cx, cy)` [m] of the 2D design region.
    size : sequence of float
        Extent `(sx, sy)` [m] of the 2D design region.
    shape : tuple of int
        Number of design pixels `(nx, ny)`.
    eps_min, eps_max : float
        Minimum and maximum relative permittivities.
    p : float, optional
        SIMP penalization exponent (`>= 1`). Default 1.0.
    """

    def __init__(self, center, size, shape, eps_min, eps_max, p=1.0):
        self.center = np.asarray(center, dtype=float)[:2]
        self.size = np.asarray(size, dtype=float)[:2]
        if np.any(self.size <= 0.0):
            raise ValueError("Design region `size` must have positive 2D extents.")
        self.shape = tuple(int(n) for n in shape[:2])
        if len(self.shape) != 2 or any(n < 1 for n in self.shape):
            raise ValueError("`shape` must be a 2-tuple of positive integers `(nx, ny)`.")
        self.eps_min = float(eps_min)
        self.eps_max = float(eps_max)
        self.p = float(p)
        self._bound_solver = None

    @property
    def bounds(self):
        """`(lo, hi)` 2D corners [m] of the design region."""
        return self.center - 0.5 * self.size, self.center + 0.5 * self.size

    def permittivity(self, rho):
        """Relative permittivity `eps(rho)` for pixel densities `rho`."""
        rho_arr = np.asarray(rho, dtype=float).reshape(self.shape)
        return self.eps_min + (self.eps_max - self.eps_min) * (rho_arr**self.p)

    def d_permittivity(self, rho):
        """Derivative `d eps / d rho` for pixel densities `rho`."""
        rho_arr = np.asarray(rho, dtype=float).reshape(self.shape)
        if self.p == 1.0:
            return np.full(self.shape, self.eps_max - self.eps_min, dtype=float)
        return self.p * (self.eps_max - self.eps_min) * (rho_arr ** (self.p - 1.0))

    def bind(self, solver):
        """
        Precompute exact overlap matrices between the `(nx, ny)` design grid
        and `solver`'s active unknowns (nodes in TM, edges in TE).
        """
        if self._bound_solver is solver:
            return
        self._bound_solver = solver
        m = solver.mesh
        lo, hi = self.bounds
        nx, ny = self.shape
        x_edges = np.linspace(lo[0], hi[0], nx + 1)
        y_edges = np.linspace(lo[1], hi[1], ny + 1)
        px0, px1 = x_edges[:-1], x_edges[1:]
        py0, py1 = y_edges[:-1], y_edges[1:]
        PX0, PY0 = np.meshgrid(px0, py0, indexing="ij")
        PX1, PY1 = np.meshgrid(px1, py1, indexing="ij")
        px0_f, px1_f = PX0.ravel(), PX1.ravel()
        py0_f, py1_f = PY0.ravel(), PY1.ravel()
        n_pix = nx * ny

        if solver.pol == "TM":
            c = m.cells
            cx0, cx1, cy0, cy1 = c["x0"], c["x1"], c["y0"], c["y1"]
            dx, dy = 0.5 * (cx1 - cx0), 0.5 * (cy1 - cy0)
            rows, cols, vals = [], [], []
            for q_idx, (ox, oy) in enumerate(((0, 0), (1, 0), (0, 1), (1, 1))):
                qx0 = cx0 + ox * dx
                qx1 = qx0 + dx
                qy0 = cy0 + oy * dy
                qy1 = qy0 + dy
                in_box = (
                    (qx1 > lo[0])
                    & (qx0 < hi[0])
                    & (qy1 > lo[1])
                    & (qy0 < hi[1])
                )
                cells_in = np.flatnonzero(in_box)
                if cells_in.size == 0:
                    continue
                nodes_q = m.cell_nodes[cells_in, q_idx]
                ox_len = np.maximum(
                    0.0,
                    np.minimum(qx1[cells_in, None], px1_f[None, :])
                    - np.maximum(qx0[cells_in, None], px0_f[None, :]),
                )
                oy_len = np.maximum(
                    0.0,
                    np.minimum(qy1[cells_in, None], py1_f[None, :])
                    - np.maximum(qy0[cells_in, None], py0_f[None, :]),
                )
                area_ov = ox_len * oy_len
                ci, pi = np.nonzero(area_ov > 0.0)
                if ci.size:
                    rows.append(nodes_q[ci])
                    cols.append(pi)
                    vals.append(area_ov[ci, pi])
            if rows:
                M_all = sp.csr_matrix(
                    (
                        np.concatenate(vals),
                        (np.concatenate(rows), np.concatenate(cols)),
                    ),
                    shape=(m.n_nodes, n_pix),
                )
            else:
                M_all = sp.csr_matrix((m.n_nodes, n_pix), dtype=float)
            M_dof = (solver._Pk_abs.T @ M_all).tocsr()
            active = np.flatnonzero(np.diff(M_dof.indptr) > 0)
            self._active_idx = active
            self._M_sub = M_dof[active].tocsr()
            self._area_active = np.asarray(self._M_sub.sum(axis=1)).ravel()
            # Baseline D with eps = eps_min in the design region
            self._D_base = solver.D.copy()
            # Current permittivity in the design region from solver's initial geometry
            self._D_unperturbed = solver.D[active] - EPS0 * (
                self._M_sub @ np.full(n_pix, solver.geo.background_eps)
            )
            # Check whether design region has any custom shape material or background
            # Recompute exact base D by subtracting the initial eps contribution on the overlap
            q_eps = solver._q["eps"]
            q_area = np.repeat(solver._q["area"][:, None], 4, axis=1)
            eps_quad = q_eps / np.maximum(q_area, 1e-300)
            # To allow arbitrary base geometry outside the design box, subtract the
            # initial permittivity integral inside the design box
            init_ov = np.zeros(m.n_nodes)
            for q_idx, (ox, oy) in enumerate(((0, 0), (1, 0), (0, 1), (1, 1))):
                qx0 = cx0 + ox * dx
                qx1 = qx0 + dx
                qy0 = cy0 + oy * dy
                qy1 = qy0 + dy
                ox_box = np.maximum(0.0, np.minimum(qx1, hi[0]) - np.maximum(qx0, lo[0]))
                oy_box = np.maximum(0.0, np.minimum(qy1, hi[1]) - np.maximum(qy0, lo[1]))
                a_box = ox_box * oy_box
                np.add.at(init_ov, m.cell_nodes[:, q_idx], EPS0 * eps_quad[:, q_idx] * a_box)
            self._D_base_active = (
                solver.D[active] - np.asarray((solver._Pk_abs.T @ init_ov)[active]).ravel()
            )
        else:
            # TE mode: each live edge e has dual cell centered at em[e] with extent (dx_e, dy_e)
            em = m.edge_mid[solver.edge_idx]
            edir = m.edge_dir[solver.edge_idx]
            L = solver.L
            Lst = solver.Lst
            dx_e = np.where(edir == 0, L, Lst)
            dy_e = np.where(edir == 0, Lst, L)
            ex0, ex1 = em[:, 0] - 0.5 * dx_e, em[:, 0] + 0.5 * dx_e
            ey0, ey1 = em[:, 1] - 0.5 * dy_e, em[:, 1] + 0.5 * dy_e
            area_e = np.maximum(L * Lst, 1e-300)

            in_box = (ex1 > lo[0]) & (ex0 < hi[0]) & (ey1 > lo[1]) & (ey0 < hi[1])
            edges_in = np.flatnonzero(in_box)
            if edges_in.size:
                ox_len = np.maximum(
                    0.0,
                    np.minimum(ex1[edges_in, None], px1_f[None, :])
                    - np.maximum(ex0[edges_in, None], px0_f[None, :]),
                )
                oy_len = np.maximum(
                    0.0,
                    np.minimum(ey1[edges_in, None], py1_f[None, :])
                    - np.maximum(ey0[edges_in, None], py0_f[None, :]),
                )
                frac_ov = (ox_len * oy_len) / area_e[edges_in, None]
                ei, pi = np.nonzero(frac_ov > 0.0)
                M_edge = sp.csr_matrix(
                    (frac_ov[ei, pi], (edges_in[ei], pi)),
                    shape=(solver.n_edges, n_pix),
                )
            else:
                M_edge = sp.csr_matrix((solver.n_edges, n_pix), dtype=float)
            active = np.flatnonzero(np.diff(M_edge.indptr) > 0)
            self._active_idx = active
            self._M_sub = M_edge[active].tocsr()
            self._frac_active = np.asarray(self._M_sub.sum(axis=1)).ravel()
            self._W_base = solver.W.copy()
            # Effective initial permittivity of each active edge: W_e = Lst / (EPS0 * L * eps_e)
            eps_init = Lst[active] / (EPS0 * L[active] * solver.W[active])
            self._eps_outside = eps_init * (1.0 - self._frac_active)
            self._scale_W = Lst[active] / (EPS0 * L[active])

    def apply(self, solver, rho):
        """Update `solver.D` (TM) or `solver.W` (TE) from pixel densities `rho`."""
        self.bind(solver)
        eps_pix = self.permittivity(rho).ravel()
        active = self._active_idx
        if active.size == 0:
            return
        if solver.pol == "TM":
            solver.D = self._D_base.copy()
            solver.D[active] = self._D_base_active + EPS0 * (self._M_sub @ eps_pix)
        else:
            solver.W = self._W_base.copy()
            eps_eff = self._eps_outside + (self._M_sub @ eps_pix)
            solver._eps_eff_active = eps_eff
            solver.W[active] = self._scale_W / eps_eff
        if solver.pml_nodes.size:
            pn = solver.pml_nodes
            Kx = solver.K[solver._x_edges]
            solver._BWx_p = (
                sp.diags(1.0 / solver.D[pn])
                @ Kx.conj().T.tocsr()[pn]
                @ sp.diags(solver.W[solver._x_edges])
            ).tocsr()

    def vjp_from_solver(self, solver, dL_dparam, rho):
        """
        Map sensitivity `dL/dD[active]` (TM) or `dL/dW[active]` (TE) back to
        `dL/drho` of shape `self.shape`.
        """
        self.bind(solver)
        active = self._active_idx
        if active.size == 0:
            return np.zeros(self.shape, dtype=float)
        if solver.pol == "TM":
            dL_deps = EPS0 * (self._M_sub.T @ dL_dparam)
        else:
            eps_eff = solver._eps_eff_active
            dW_deps = -solver.W[active] / eps_eff
            dL_deps = self._M_sub.T @ (dW_deps * dL_dparam)
        return dL_deps.reshape(self.shape) * self.d_permittivity(rho)


class _ModeReadout:
    """Linear readout wrapper for `SubgridModeMonitor`."""

    def __init__(self, solver, monitor, mode_index=0):
        self.monitor = monitor
        self.mode_index = int(mode_index)
        if monitor.is_first_update:
            monitor._initialize(solver)
            monitor.is_first_update = False
        if monitor.modes is None:
            monitor._solve_modes()
        self.freqs = monitor.frequencies
        n_f = self.freqs.size
        M1 = monitor.line.n_seg + 1
        n_ec = monitor._edge_cols.size
        pol = solver.pol

        # Build linear weights (n_f, M1) for u and (n_f, n_ec) for g_edge
        wu_plus = np.zeros((n_f, M1), dtype=complex)
        wu_minus = np.zeros((n_f, M1), dtype=complex)
        wg_line_plus = np.zeros((n_f, M1), dtype=complex)
        wg_line_minus = np.zeros((n_f, M1), dtype=complex)

        for i, modes in enumerate(monitor.modes):
            m = modes[self.mode_index]
            if pol == "TE":
                # E_x_h = sum(g_a * conj(m.u)), e_x_H = sum(conj(m.g_a_mean) * u)
                wg_line_plus[i] = 0.25 * np.conj(m.u)
                wg_line_minus[i] = 0.25 * np.conj(m.u)
                wu_plus[i] = 0.25 * np.conj(m.g_a_mean)
                wu_minus[i] = -0.25 * np.conj(m.g_a_mean)
            else:
                # E_x_h = sum(u * conj(m.g_a_mean)), e_x_H = sum(conj(m.u) * g_a)
                wu_plus[i] = 0.25 * np.conj(m.g_a_mean)
                wu_minus[i] = 0.25 * np.conj(m.g_a_mean)
                wg_line_plus[i] = 0.25 * np.conj(m.u)
                wg_line_minus[i] = -0.25 * np.conj(m.u)

        # Map line nodes (M1) to solver._dofs ( masked ) and edge_cols
        self.dofs = monitor._dofs
        self.dof_mask = monitor._dof_mask
        self.edge_cols = monitor._edge_cols
        self.wu_plus = wu_plus * self.dof_mask[None, :]
        self.wu_minus = wu_minus * self.dof_mask[None, :]
        self.wg_plus = (monitor._G_sub.T @ wg_line_plus.T).T if n_ec else np.zeros((n_f, 0), complex)
        self.wg_minus = (monitor._G_sub.T @ wg_line_minus.T).T if n_ec else np.zeros((n_f, 0), complex)

    def shapes(self):
        n_f = self.freqs.size
        return {"a_plus": (n_f,), "a_minus": (n_f,)}

    def reset(self):
        self.monitor._u_acc.fill(0.0)
        self.monitor._g_edge_acc.fill(0.0)

    def outputs(self):
        ap, am = self.monitor.get_amplitudes(mode_index=self.mode_index)
        return {"a_plus": ap, "a_minus": am}

    def prepare_bwd(self, G_mon):
        gp = np.broadcast_to(
            np.atleast_1d(np.asarray(G_mon.get("a_plus", 0.0), dtype=complex)),
            (self.freqs.size,),
        )
        gm = np.broadcast_to(
            np.atleast_1d(np.asarray(G_mon.get("a_minus", 0.0), dtype=complex)),
            (self.freqs.size,),
        )
        self._cu = gp[:, None] * np.conj(self.wu_plus) + gm[:, None] * np.conj(
            self.wu_minus
        )  # (n_f, M1)
        self._cg = gp[:, None] * np.conj(self.wg_plus) + gm[:, None] * np.conj(
            self.wg_minus
        )  # (n_f, n_ec)

    def inject_bwd(self, n, dt, ub1, gb):
        t = n * dt
        phase = 2j * np.pi * self.freqs
        ku = dt * np.exp(phase * (t + dt))
        kg = dt * np.exp(phase * (t + 0.5 * dt))
        du = np.real(ku @ self._cu)
        np.add.at(ub1, self.dofs, du)
        if self.edge_cols.size:
            dg = np.real(kg @ self._cg)
            gb[self.edge_cols] += dg


class _DFTReadout:
    """Linear readout wrapper for `SubgridDFTMonitor`."""

    def __init__(self, solver, monitor):
        self.monitor = monitor
        if monitor.is_first_update:
            monitor._initialize(solver)
            monitor.is_first_update = False
        self.freqs = monitor.frequencies
        self.pol = solver.pol

    def shapes(self):
        n_f = self.freqs.size
        return {c: (n_f, *self.monitor.out_shape) for c in self.monitor.components}

    def reset(self):
        if self.monitor._g_acc is not None:
            self.monitor._g_acc.fill(0.0)
        if self.monitor._u_acc is not None:
            self.monitor._u_acc.fill(0.0)

    def outputs(self):
        return {c: self.monitor.get_field(c)[0] for c in self.monitor.components}

    def prepare_bwd(self, G_mon):
        mon = self.monitor
        n_f = self.freqs.size
        self._cu = (
            np.zeros((n_f, mon._dof_cols.size), dtype=complex)
            if mon._dof_cols is not None
            else None
        )
        self._cg = (
            np.zeros((n_f, mon._edge_cols.size), dtype=complex)
            if mon._edge_cols is not None
            else None
        )
        for comp, ct in G_mon.items():
            ct_flat = np.asarray(ct, dtype=complex).reshape(n_f, -1)
            if comp in ("Hz", "Ez", "u"):
                self._cu += (mon._S_u.conj().T @ ct_flat.T).T
            elif self.pol == "TE":
                if comp == "Ex":
                    self._cg += (mon._Mx.conj().T @ ct_flat.T).T
                elif comp == "Ey":
                    self._cg += (mon._My.conj().T @ ct_flat.T).T
            else:
                if comp == "Hx":
                    self._cg -= (mon._Mx.conj().T @ ct_flat.T).T
                elif comp == "Hy":
                    self._cg -= (mon._My.conj().T @ ct_flat.T).T

    def inject_bwd(self, n, dt, ub1, gb):
        mon = self.monitor
        t = n * dt
        phase = 2j * np.pi * self.freqs
        if self._cu is not None and mon._dof_cols.size:
            ku = dt * np.exp(phase * (t + dt))
            ub1[mon._dof_cols] += np.real(ku @ self._cu)
        if self._cg is not None and mon._edge_cols.size:
            kg = dt * np.exp(phase * (t + 0.5 * dt))
            gb[mon._edge_cols] += np.real(kg @ self._cg)


class SubgridAdjointSimulation:
    """
    Differentiable time-domain simulation on `SubgridFIT2D`, compatible with
    `wakis.inverse.jax_simulation`.

    Parameters
    ----------
    solver : SubgridFIT2D
        Configured subgrid solver (with PML and soft current sources registered).
    design : SubgridMaterialGrid
        2D parameterized design region.
    monitors : dict of {str: SubgridModeMonitor or SubgridDFTMonitor}
        Named frequency-domain monitors.
    n_steps : int
        Number of time steps to run.
    mode_index : int, optional
        Mode index for `SubgridModeMonitor` readouts. Default 0.
    """

    def __init__(self, solver, design, monitors, n_steps, mode_index=0):
        self.solver = solver
        self.design = design
        self.monitors = dict(monitors)
        self.n_steps = int(n_steps)
        self.mode_index = int(mode_index)
        self.design.bind(self.solver)
        self.readouts = {}
        for name, mon in self.monitors.items():
            if isinstance(mon, SubgridModeMonitor):
                self.readouts[name] = _ModeReadout(
                    self.solver, mon, mode_index=self.mode_index
                )
            elif isinstance(mon, SubgridDFTMonitor):
                self.readouts[name] = _DFTReadout(self.solver, mon)
            else:
                raise TypeError(
                    f"Unsupported monitor type {type(mon).__name__} in SubgridAdjointSimulation."
                )
        self.rho = None
        self._recorded = False
        self._trace = None

    def output_shapes(self):
        """Return `{monitor_name: {key: shape}}` as expected by `jax_simulation`."""
        return {name: ro.shapes() for name, ro in self.readouts.items()}

    def forward(self, rho, record=True):
        """
        Run the forward simulation for design densities `rho` and return monitor
        outputs `{monitor_name: {key: complex_ndarray}}`.
        """
        rho_arr = np.asarray(rho, dtype=float).reshape(self.design.shape).copy()
        self.design.apply(self.solver, rho_arr)
        self.solver.reset()
        for ro in self.readouts.values():
            ro.reset()

        active = self.design._active_idx
        N = self.n_steps
        dt = self.solver.dt
        pol = self.solver.pol
        if record:
            self._trace = np.zeros((N, active.size), dtype=self.solver._dtype)
        else:
            self._trace = None

        for n in range(N):
            t = n * dt
            u_prev_act = self.solver.u[active].copy() if (record and pol == "TM") else None
            self.solver.one_step()
            if record:
                if pol == "TM":
                    self._trace[n] = (self.solver.u[active] - u_prev_act) / dt
                else:
                    self._trace[n] = self.solver.f[active]
            for mon in self.monitors.values():
                mon.update(self.solver, t)

        self.rho = rho_arr
        self._recorded = bool(record)
        return {name: ro.outputs() for name, ro in self.readouts.items()}

    def vjp(self, G):
        """
        Reverse-mode vector-Jacobian product returning `dL / drho` of shape
        `self.design.shape` for output cotangents `G` (where
        `dL = Re sum(G * output)`).
        """
        if not self._recorded or self._trace is None:
            self.forward(self.rho, record=True)

        solver = self.solver
        dt = solver.dt
        N = self.n_steps
        pol = solver.pol
        pn = solver.pml_nodes
        sn = solver._sig_nodes
        active = self.design._active_idx

        for name, ro in self.readouts.items():
            ro.prepare_bwd(G.get(name, {}))

        ub = np.zeros(solver.n_dof, dtype=solver._dtype)
        fb = np.zeros(solver.n_edges, dtype=solver._dtype)
        uxb = np.zeros(pn.size, dtype=solver._dtype)
        uyb = np.zeros(pn.size, dtype=solver._dtype)
        dL_dparam = np.zeros(active.size, dtype=float)

        for n in range(N - 1, -1, -1):
            ub1 = ub.copy()
            fb_p = fb.copy()
            gb_mon = np.zeros(solver.n_edges, dtype=solver._dtype)
            for ro in self.readouts.values():
                ro.inject_bwd(n, dt, ub1, gb_mon)

            # Transpose of node & PML split update (Step 4)
            if pn.size:
                uxb_tot = uxb + ub1[pn]
                uyb_tot = uyb + ub1[pn]
                ub1[pn] = 0.0
                uxb = solver._axn * uxb_tot
                uyb = solver._ayn * uyb_tot
                rxb = solver._bxn * uxb_tot
                ryb = solver._byn * uyb_tot
                rxb = rxb - ryb
            rb = dt * ub1
            ub = ub1.copy()
            if sn.size:
                rb[sn] = solver._bsn * ub1[sn]
                ub[sn] = solver._asn * ub1[sn]
            if pn.size:
                rb[pn] = ryb
                fb_p[solver._x_edges] = fb_p[solver._x_edges] + (
                    solver._BWx_p.conj().T @ rxb
                )

            # Transpose of r = r0 / D (Step 3)
            r0b = rb / solver.D
            gb_tot = (solver.K @ r0b) + gb_mon

            # Accumulate sensitivity w.r.t. D (TM) or W (TE)
            if pol == "TM":
                # r_n = (u^{n+1} - u^n) / dt = r0_n / D -> d(u^{n+1})/dD = -dt * r_n / D
                dL_dparam -= np.real(r0b[active] * np.conj(self._trace[n]))
            else:
                # g = W * f1 -> dg/dW = f1
                dL_dparam += np.real(gb_tot[active] * np.conj(self._trace[n]))

            # Transpose of g = W * f1 and edge update (Steps 2 & 1)
            fb_p = fb_p + solver._apply_W(gb_tot)
            fb = solver._ae_full * fb_p
            z = solver.KH @ (solver._be_full * fb_p)
            if solver.lts is not None:
                z = solver.lts.apply_T(z)
            ub = ub - z

        return self.design.vjp_from_solver(solver, dL_dparam, self.rho)


def subgrid_mode_amplitudes(port_output):
    """
    Identity helper returning `(port_output['a_plus'], port_output['a_minus'])`
    from a `SubgridModeMonitor` output dict in `SubgridAdjointSimulation` or
    `jax_simulation`.
    """
    return port_output["a_plus"], port_output["a_minus"]
