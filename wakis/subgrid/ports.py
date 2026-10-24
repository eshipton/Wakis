# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Polarization mapping, 1D waveguide eigenmode solver, and unidirectional mode
injection for `SubgridFIT2D`.

Polarization mapping between `PICSimulation(..., two_d=True)` and `SubgridFIT2D`
--------------------------------------------------------------------------------
In `wakis.templates.photonics.PICSimulation` (`two_d=True`, 2D effective-index
method):

1. The vertical SOI slab stack (`z in [z_min, z_max]`) is reduced to its
   fundamental horizontal-E slab mode (`polarization='TE'`, dominant `E_y(z)`
   tangential to the horizontal slab faces) with effective index `n_eff_slab`.
2. In the 2D `(x, y)` calculation (`Nz = 1`, periodic `z` so `d/dz = 0`),
   `PICSimulation.build()` sets `solver.ieps[:, :, :, 'z'] = 0`, which freezes
   `E_z = 0` and restricts `SolverFIT3D` to the **in-plane electric field
   `(E_x, E_y)` and out-of-plane magnetic field `H_z`**:

   .. math::

       \\mu_0 \\frac{\\partial H_z}{\\partial t}
           = \\frac{\\partial E_x}{\\partial y}
           - \\frac{\\partial E_y}{\\partial x} - M_z, \\qquad
       \\varepsilon_0 \\varepsilon_x \\frac{\\partial E_x}{\\partial t}
           = \\frac{\\partial H_z}{\\partial y} - J_x, \\qquad
       \\varepsilon_0 \\varepsilon_y \\frac{\\partial E_y}{\\partial t}
           = -\\frac{\\partial H_z}{\\partial x} - J_y.

In `SubgridFIT2D`, this exact system is **`polarization="TE"`** (`TE_z`):

* Nodal scalar unknown (at mesh corners, integer time levels `t^n = n dt`):
  `u = H_z` [A/m], with dual-cell mass `D_i = (P^T m)_i`, `m_a = mu_0 A_a^*`.
* Edge flux unknown (on primal cell sides, half-integer time levels
  `t^{n+1/2} = (n + 1/2) dt`): `f_e = L_e eps_0 eps_e (t_e x E)_z` [C/m], and
  dual-edge voltage `g_e = W_e f_e = L_e^* (t_e x E)_z` [V]:

  - on `x`-directed edges (`t_e = +x`): `(t_e x E)_z = +E_y`, so
    `E_y = +g_x / L_y^*`;
  - on `y`-directed edges (`t_e = +y`): `(t_e x E)_z = -E_x`, so
    `E_x = -g_y / L_x^*`.

* Sub-pixel material averaging in `SubgridFIT2D(polarization="TE")` integrates
  `R_k = int dn / eps` in series across the dual edge (normal to the primal
  edge, i.e. along the `E` component `(t_e x E)_z`) and averages `R_k^-1` in
  parallel along the primal edge. For a sidewall normal to `y`, `E_y` (on
  `x`-edges) sees the harmonic mean of `eps` across `y` while `E_x` (on
  `y`-edges) sees the arithmetic mean along `y`, matching Kottke's subpixel
  smoothing in `PICSimulation`.
"""

import numpy as np
from scipy.constants import c as c_light
from scipy.linalg import eigh

__all__ = [
    "PIC_POLARIZATION",
    "polarization_mapping",
    "SubgridMode",
    "solve_port_modes_2d",
    "SubgridModeSource",
]

PIC_POLARIZATION = "TE"


def polarization_mapping():
    """
    Return the correspondence between `PICSimulation(two_d=True)` and
    `SubgridFIT2D`.
    """
    return {
        "subgrid_polarization": PIC_POLARIZATION,
        "node_unknown_u": "Hz",
        "x_edge_flux_f": "+Lx * eps_0 * eps_y * Ey",
        "y_edge_flux_f": "-Ly * eps_0 * eps_x * Ex",
        "x_edge_voltage_g": "+Ly_dual * Ey",
        "y_edge_voltage_g": "-Lx_dual * Ex",
        "cartesian_2d_eim_components": ("Ex", "Ey", "Hz"),
        "slab_stack_polarization": "TE",
    }


class _PortLine2D:
    """
    Discrete 1D cross-section of a `SubgridFIT2D` mesh along a port line
    normal to axis `a` (`0` for x, `1` for y) and spanning axis `b = 1 - a`.
    """

    def __init__(self, solver, center, size):
        center = np.asarray(center, dtype=float)
        size = np.asarray(size, dtype=float)
        zero_axes = np.flatnonzero(size[:2] == 0.0)
        if zero_axes.size != 1:
            raise ValueError(
                "Port `size` must have exactly one zero entry in (sx, sy)."
            )
        self.solver = solver
        self.mesh = mesh = solver.mesh
        self.a = a = int(zero_axes[0])
        self.b = b = 1 - a

        coarse_a = mesh.xc if a == 0 else mesh.yc
        coarse_b = mesh.yc if a == 0 else mesh.xc
        lo_b = (
            float(coarse_b[0])
            if not np.isfinite(size[b])
            else max(float(center[b] - 0.5 * size[b]), float(coarse_b[0]))
        )
        hi_b = (
            float(coarse_b[-1])
            if not np.isfinite(size[b])
            else min(float(center[b] + 0.5 * size[b]), float(coarse_b[-1]))
        )
        self.lo_b, self.hi_b = lo_b, hi_b

        inner_coarse = coarse_a[1:-1]
        k_c = int(np.argmin(np.abs(inner_coarse - center[a])))
        pos_a = float(inner_coarse[k_c])
        self.position = pos_a

        domain_span_a = float(coarse_a[-1] - coarse_a[0])
        domain_span_b = float(coarse_b[-1] - coarse_b[0])
        tol_a = 1e-9 * domain_span_a
        tol_b = 1e-9 * domain_span_b

        edir = mesh.edge_dir[solver.edge_idx]
        em = mesh.edge_mid[solver.edge_idx]
        mesh_edges = mesh.edges[solver.edge_idx]

        # Transverse edges lying on coord_a == pos_a within [lo_b, hi_b]
        on_line_t = (
            (edir == b)
            & (np.abs(em[:, a] - pos_a) <= tol_a)
            & (em[:, b] >= lo_b - tol_b)
            & (em[:, b] <= hi_b + tol_b)
        )
        t_edges = np.flatnonzero(on_line_t)
        if t_edges.size < 2:
            raise ValueError("Port line has too few transverse edges.")
        order = np.argsort(em[t_edges, b])
        t_edges = t_edges[order]
        self.t_edges = t_edges
        self.n_seg = M = t_edges.size

        line_nodes = np.concatenate(
            [mesh_edges[t_edges, 0], [mesh_edges[t_edges[-1], 1]]]
        )
        self.line_nodes = line_nodes
        self.node_b = mesh.nodes[line_nodes, b]
        self.mid_b = em[t_edges, b]
        self.L_b = solver.L[t_edges]

        # Dual transverse widths at the M + 1 line nodes
        db_dual = np.zeros(M + 1, dtype=float)
        db_dual[:-1] += 0.5 * self.L_b
        db_dual[1:] += 0.5 * self.L_b
        self.db_dual = db_dual

        # Map line nodes to solver DOFs
        node_to_dof = -np.ones(mesh.n_nodes, dtype=int)
        node_to_dof[solver.dof] = np.arange(solver.n_dof)
        self.dofs = node_to_dof[line_nodes]

        # Normal edges (edge_dir == a) touching each line node on the low (-a)
        # and high (+a) sides. A transition node on a coarse/fine interface
        # parallel to `a` owns two normal edges on each side (one coarse
        # half-strip and one fine half-strip).
        node_to_line = -np.ones(mesh.n_nodes, dtype=int)
        node_to_line[line_nodes] = np.arange(M + 1)

        n_cand = np.flatnonzero(edir == a)
        j_lo = node_to_line[mesh_edges[n_cand, 1]]
        j_hi = node_to_line[mesh_edges[n_cand, 0]]

        self.e_lo_groups = [[] for _ in range(M + 1)]
        self.e_hi_groups = [[] for _ in range(M + 1)]
        for e_idx, jl in zip(n_cand[j_lo >= 0], j_lo[j_lo >= 0]):
            self.e_lo_groups[int(jl)].append(int(e_idx))
        for e_idx, jh in zip(n_cand[j_hi >= 0], j_hi[j_hi >= 0]):
            self.e_hi_groups[int(jh)].append(int(e_idx))

        has_lo = np.array([len(g) > 0 for g in self.e_lo_groups], dtype=bool)
        has_hi = np.array([len(g) > 0 for g in self.e_hi_groups], dtype=bool)
        valid = (self.dofs >= 0) & has_lo & has_hi
        valid[0] = False
        valid[-1] = False
        self.active = np.flatnonzero(valid)
        if self.active.size < 2:
            raise ValueError("Port line has too few active interior nodes.")

        self.L_a_lo = np.zeros(M + 1, dtype=float)
        self.L_a_hi = np.zeros(M + 1, dtype=float)
        self.w_a_lo_1d = np.zeros(M + 1, dtype=float)
        self.w_a_hi_1d = np.zeros(M + 1, dtype=float)
        dual_area_node = np.zeros(M + 1, dtype=float)

        for j in range(M + 1):
            if self.e_lo_groups[j]:
                el = np.asarray(self.e_lo_groups[j], dtype=int)
                wt_l = solver.Lst[el]
                self.L_a_lo[j] = float(
                    np.sum(solver.L[el] * wt_l) / np.maximum(np.sum(wt_l), 1e-30)
                )
                self.w_a_lo_1d[j] = float(np.sum(solver.L[el] * solver.W[el]))
                dual_area_node[j] += 0.5 * float(np.sum(solver.L[el] * wt_l))
            if self.e_hi_groups[j]:
                eh = np.asarray(self.e_hi_groups[j], dtype=int)
                wt_h = solver.Lst[eh]
                self.L_a_hi[j] = float(
                    np.sum(solver.L[eh] * wt_h) / np.maximum(np.sum(wt_h), 1e-30)
                )
                self.w_a_hi_1d[j] = float(np.sum(solver.L[eh] * solver.W[eh]))
                dual_area_node[j] += 0.5 * float(np.sum(solver.L[eh] * wt_h))

        self.L_a = 0.5 * (self.L_a_lo + self.L_a_hi)
        self.w_a_1d = 0.5 * (self.w_a_lo_1d + self.w_a_hi_1d)

        dof_valid = np.maximum(self.dofs, 0)
        self.D_node = np.where(self.dofs >= 0, solver.D[dof_valid], 0.0)
        L_a_eff = np.where(
            db_dual > 0, dual_area_node / np.maximum(db_dual, 1e-30), self.L_a
        )
        self.m_1d = np.where(
            L_a_eff > 0, self.D_node / np.where(L_a_eff > 0, L_a_eff, 1.0), 0.0
        )
        self.w_t_1d = solver.W[t_edges] / solver.Lst[t_edges]


class SubgridMode:
    """
    Waveguide eigenmode on a 1D port cross-section of `SubgridFIT2D`,
    normalized to unit time-averaged power (`1/2 Re int (e x h^*) . a db = 1 W/m`).

    Attributes match `wakis.mode_solver.Mode` (`frequency`, `beta`,
    `beta_tilde`, `neff`, `e`, `h`, `h_mean`, `axes`, `coords`) so existing
    PIC analysis and plotting helpers work unchanged.
    """

    def __init__(self, line, frequency, beta_tilde, u_line):
        self.line = line
        self.frequency = float(frequency)
        self.beta_tilde = float(beta_tilde)

        j_peak = int(np.argmax(np.abs(u_line)))
        if u_line[j_peak] < 0:
            u_line = -u_line
        L_a_ref = float(line.L_a[j_peak])
        arg = min(1.0, max(0.0, self.beta_tilde * L_a_ref / 2.0))
        self.beta = float(2.0 / L_a_ref * np.arcsin(arg))
        self.neff = float(self.beta * c_light / (2.0 * np.pi * self.frequency))

        dt = line.solver.dt
        w_tilde = 2.0 / dt * np.sin(np.pi * self.frequency * dt)
        self.w_tilde = float(w_tilde)

        # Dual-edge voltage g_a = (beta_tilde / w_tilde) * w_a_1d * u on the
        # normal edges, and its average from the -a and +a half-steps onto the
        # port line
        g_a = (self.beta_tilde / w_tilde) * line.w_a_1d * u_line
        L_lo = np.where(line.L_a_lo > 0, line.L_a_lo, L_a_ref)
        L_hi = np.where(line.L_a_hi > 0, line.L_a_hi, L_a_ref)
        cos_mean = 0.5 * (
            np.exp(1j * self.beta * L_lo / 2.0)
            + np.exp(-1j * self.beta * L_hi / 2.0)
        )
        g_a_mean = g_a * cos_mean

        power = 0.5 * float(np.real(np.sum(g_a_mean * np.conj(u_line))))
        scale = 1.0 / np.sqrt(max(power, 1e-30))
        self.u = (u_line * scale).astype(complex)
        self.g_a = (g_a * scale).astype(complex)
        self.g_a_mean = (g_a_mean * scale).astype(complex)
        self.L_a_ref = L_a_ref

        # Populate Mode-compatible 2D arrays e['b'], e['c'], h['b'], h['c']
        n_pts = line.node_b.size
        et = self.g_a / np.maximum(line.db_dual, 1e-30)
        et_mean = self.g_a_mean / np.maximum(line.db_dual, 1e-30)
        z_zero = np.array([0.0])

        if line.a == 0:  # x-normal port: (a, b, c) = (x, y, z)
            self.axes = "xyz"
            shape = (n_pts, 1)
            zeros = np.zeros(shape, dtype=complex)
            if line.solver.pol == "TE":
                # Ey = +g_x / dy^*, Hz = u
                self.e = {"b": et.reshape(shape), "c": zeros.copy()}
                self.h = {"b": zeros.copy(), "c": self.u.reshape(shape)}
                self.h_mean = {
                    "b": zeros.copy(),
                    "c": (self.u * cos_mean).reshape(shape),
                }
            else:
                # Ez = u, Hy = -g_x / dy^*
                self.e = {"b": zeros.copy(), "c": self.u.reshape(shape)}
                self.h = {"b": (-et).reshape(shape), "c": zeros.copy()}
                self.h_mean = {"b": (-et_mean).reshape(shape), "c": zeros.copy()}
            self.coords = {
                "Eb": (line.node_b, z_zero),
                "Ec": (line.node_b, z_zero),
                "Hb": (line.node_b, z_zero),
                "Hc": (line.node_b, z_zero),
            }
        else:  # y-normal port: (a, b, c) = (y, z, x)
            self.axes = "yzx"
            shape = (1, n_pts)
            zeros = np.zeros(shape, dtype=complex)
            if line.solver.pol == "TE":
                # Ex = -g_y / dx^*, Hz = u
                self.e = {"b": zeros.copy(), "c": (-et).reshape(shape)}
                self.h = {"b": self.u.reshape(shape), "c": zeros.copy()}
                self.h_mean = {
                    "b": (self.u * cos_mean).reshape(shape),
                    "c": zeros.copy(),
                }
            else:
                # Ez = u, Hx = +g_y / dx^*
                self.e = {"b": self.u.reshape(shape), "c": zeros.copy()}
                self.h = {"b": zeros.copy(), "c": et.reshape(shape)}
                self.h_mean = {"b": zeros.copy(), "c": et_mean.reshape(shape)}
            self.coords = {
                "Eb": (z_zero, line.node_b),
                "Ec": (z_zero, line.node_b),
                "Hb": (z_zero, line.node_b),
                "Hc": (z_zero, line.node_b),
            }


def solve_port_modes_2d(solver, frequency, center, size, n_modes=1):
    """
    Solve the propagating waveguide modes of `solver` (`SubgridFIT2D`) on the
    port line defined by `(center, size)` at `frequency` [Hz].

    Eliminating the normal edge fluxes from the 2D FIT leapfrog equations on a
    cross-section uniform along the port normal `a` with
    `d/da -> -1j * beta_tilde` and `d/dt -> 1j * omega_tilde` gives the
    symmetric generalized eigenvalue problem on the port line nodes:

    .. math::

        \\left(\\tilde{\\omega}^2 \\mathbf{m}_{1\\mathrm{D}}
        - \\mathbf{G}_{1\\mathrm{D}}^T \\mathbf{w}_{t,1\\mathrm{D}}
          \\mathbf{G}_{1\\mathrm{D}}\\right) \\mathbf{u}
        = \\tilde{\\beta}^2 \\mathbf{w}_{a,1\\mathrm{D}} \\mathbf{u},

    whose eigenvectors are exact discrete eigenmodes of `SubgridFIT2D`.

    Parameters
    ----------
    solver : SubgridFIT2D
    frequency : float
        Frequency [Hz].
    center, size : sequence of float
        Port centre and extent [m]; exactly one of `size[0]`, `size[1]` is 0.
    n_modes : int, optional
        Number of modes to return, sorted by decreasing `neff`. Default 1.

    Returns
    -------
    list of SubgridMode
    """
    line = _PortLine2D(solver, center, size)
    dt = solver.dt
    w_tilde = 2.0 / dt * np.sin(np.pi * float(frequency) * dt)

    # Assemble the 1D stiffness G_1D^T diag(w_t_1d) G_1D on all M + 1 line nodes
    M = line.n_seg
    wt = line.w_t_1d
    diag_all = np.zeros(M + 1, dtype=float)
    diag_all[:-1] += wt
    diag_all[1:] += wt
    off_all = -wt

    act = line.active
    A = np.diag(w_tilde**2 * line.m_1d[act] - diag_all[act])
    for k in range(act.size - 1):
        if act[k + 1] == act[k] + 1:
            val = -off_all[act[k]]
            A[k, k + 1] = val
            A[k + 1, k] = val

    s = 1.0 / np.sqrt(np.maximum(line.w_a_1d[act], 1e-30))
    H = (s[:, None] * A) * s[None, :]
    eigvals, eigvecs = eigh(H)

    order = np.argsort(-eigvals)
    modes = []
    for idx in order:
        if len(modes) >= int(n_modes):
            break
        lam = float(eigvals[idx])
        if lam <= 0.0:
            continue
        beta_tilde = np.sqrt(lam)
        u_line = np.zeros(M + 1, dtype=float)
        u_line[act] = s * eigvecs[:, idx]
        modes.append(SubgridMode(line, frequency, beta_tilde, u_line))
    return modes


class SubgridModeSource:
    """
    Unidirectional waveguide mode source on a port line of `SubgridFIT2D`.

    Injects a unit-power `SubgridMode` in direction `direction` (`'+'` or `'-'`)
    using the discrete Total-Field/Scattered-Field (equivalence principle)
    weights on the port-line nodes (`u` update at `t + dt/2 + h_advance`) and
    the upstream normal edges (`f` update at `t`), registered via
    `solver.add_current`.

    Parameters
    ----------
    src_time : callable
        Time waveform `s(t)` (e.g. `GaussianSource`).
    center, size : sequence of float
        Port line centre and extent [m].
    direction : {'+', '-', 1, -1}, optional
        Propagation direction along the port normal. Default `'+'`.
    mode_index : int, optional
        Mode index sorted by decreasing `neff`. Default 0.
    amplitude : float, optional
        Modal amplitude [sqrt(W/m)]. Default 1.0.
    mode_frequency : float, optional
        Frequency [Hz] at which the eigenmode is solved. Defaults to
        `src_time.frequency`.
    """

    def __init__(
        self,
        src_time,
        center,
        size,
        direction="+",
        mode_index=0,
        amplitude=1.0,
        mode_frequency=None,
    ):
        self.src_time = src_time
        self.center = np.asarray(center, dtype=float)
        self.size = np.asarray(size, dtype=float)
        if isinstance(direction, str):
            if direction not in ("+", "-"):
                raise ValueError("direction must be '+' or '-'.")
            self.direction = 1 if direction == "+" else -1
        else:
            if int(direction) not in (1, -1):
                raise ValueError("direction must be +1 or -1.")
            self.direction = int(direction)
        self.mode_index = int(mode_index)
        self.amplitude = float(amplitude)
        self.mode_frequency = (
            float(src_time.frequency)
            if mode_frequency is None
            else float(mode_frequency)
        )
        self.end_time = getattr(src_time, "end_time", np.inf)
        self.mode = None
        self.current_ids = []

    def register(self, solver):
        """Solve the port mode and register its TF/SF currents on `solver`."""
        modes = solve_port_modes_2d(
            solver,
            self.mode_frequency,
            self.center,
            self.size,
            n_modes=self.mode_index + 1,
        )
        if len(modes) <= self.mode_index:
            raise ValueError(
                f"Mode {self.mode_index} does not propagate at "
                f"{self.mode_frequency:.4g} Hz."
            )
        self.mode = mode = modes[self.mode_index]
        line = mode.line
        act = line.active
        d = self.direction

        # Upstream normal edges: e_lo_groups for d = +1, e_hi_groups for d = -1
        groups_up = line.e_lo_groups if d > 0 else line.e_hi_groups
        w_a_up = line.w_a_lo_1d[act] if d > 0 else line.w_a_hi_1d[act]
        dofs_act = line.dofs[act]

        u_act = np.real(mode.u[act])
        edges_up, w_f = [], []
        for j, u_val in zip(act, u_act):
            for e_idx in groups_up[int(j)]:
                edges_up.append(e_idx)
                w_f.append(d * self.amplitude * u_val)
        edges_up = np.asarray(edges_up, dtype=int)
        w_f = np.asarray(w_f, dtype=float)

        w_u = (
            self.amplitude
            * (mode.beta_tilde / mode.w_tilde)
            * w_a_up
            * u_act
        )

        omega = 2.0 * np.pi * self.mode_frequency
        h_advance = float(mode.beta * mode.L_a_ref / (2.0 * omega))

        id_f = solver.add_current(
            edges_up, w_f, waveform=self.src_time, kind="edge", time_offset=0.0
        )
        id_u = solver.add_current(
            dofs_act,
            w_u,
            waveform=self.src_time,
            kind="node",
            time_offset=h_advance,
        )
        self.current_ids = [id_f, id_u]
        return self
