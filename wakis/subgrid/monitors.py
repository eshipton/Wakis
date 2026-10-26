# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Frequency-domain field, mode-overlap, and Poynting-flux monitors for
`SubgridFIT2D`.

* `SubgridDFTMonitor`: running discrete Fourier transform of `(Ex, Ey, Hz)`
  (in TE) or `(Ez, Hx, Hy)` (in TM) at arbitrary points, lines, or 2D grids
  via `SubgridFIT2D.edge_field_matrices` and `SubgridFIT2D.interpolation`;
* `SubgridModeMonitor`: forward and backward modal amplitudes `(a+, a-)` on a
  port line, normalized identically to `wakis.mode_solver.ModeMonitor`;
* `SubgridFluxMonitor`: Poynting flux spectrum `Phi(f)` through a line or out
  of a 2D box, normalized identically to `wakis.field_monitors.FluxMonitor`.
"""

import numpy as np
import scipy.sparse as sp
from scipy.constants import c as C0
from scipy.constants import epsilon_0 as EPS0
from scipy.constants import mu_0 as MU0

from ..resonances import ResonanceMonitor as SubgridResonanceMonitor
from .ports import _PortLine2D, solve_port_modes_2d

__all__ = [
    "SubgridDFTMonitor",
    "SubgridModeMonitor",
    "SubgridFluxMonitor",
    "SubgridNear2FarMonitor",
    "Near2FarMonitor2D",
    "SubgridDiffractionMonitor",
    "SubgridResonanceMonitor",
]


def _restrict_sparse_columns(mat):
    """
    Restrict a CSR matrix `(n_rows, n_cols)` to its non-empty columns,
    returning `(mat_sub, active_cols)`.
    """
    mat = mat.tocsr()
    cols = np.unique(mat.indices)
    if cols.size == 0:
        return sp.csr_matrix((mat.shape[0], 0), dtype=mat.dtype), np.array([], dtype=int)
    remap = np.zeros(mat.shape[1], dtype=int)
    remap[cols] = np.arange(cols.size)
    sub = sp.csr_matrix(
        (mat.data.copy(), remap[mat.indices], mat.indptr.copy()),
        shape=(mat.shape[0], cols.size),
    )
    return sub, cols


class SubgridDFTMonitor:
    """
    Running discrete Fourier transform of 2D field components on points, lines,
    or a 2D rectilinear grid `(x_1d, y_1d)` in `SubgridFIT2D`.

    Uses `solver.edge_field_matrices` for the in-plane edge fields (`Ex`, `Ey`
    in TE; `Hx`, `Hy` in TM) at half-integer time levels `t + dt/2`, and
    `solver.interpolation` for the nodal scalar field (`Hz` in TE; `Ez` in TM)
    at integer time levels `t + dt`.

    Parameters
    ----------
    frequencies : array_like
        Frequencies to monitor [Hz].
    points : array_like or tuple (x_1d, y_1d), optional
        Either an `(n_pts, 2)` array of 2D coordinates [m], or a 2-tuple of 1D
        coordinate arrays `(x_1d, y_1d)` defining a Cartesian sampling grid of
        shape `(len(x_1d), len(y_1d))`. If omitted, `center` and `size` define
        a point or line.
    components : sequence of str, optional
        Field components, e.g. `('Ex', 'Ey', 'Hz')`. Default `('Ex', 'Ey', 'Hz')`.
    center, size : sequence of float, optional
        Fallback region specification [m] when `points` is None.
    n_pts : int, optional
        Number of sample points along a line when specified via `center`/`size`.
        Default 101.
    """

    def __init__(
        self,
        frequencies,
        points=None,
        components=("Ex", "Ey", "Hz"),
        center=(0.0, 0.0),
        size=(0.0, 0.0),
        n_pts=101,
    ):
        self.frequencies = np.atleast_1d(np.asarray(frequencies, dtype=float))
        self.components = tuple(components)
        self.grid_coords = None

        if points is not None:
            if (
                isinstance(points, (tuple, list))
                and len(points) == 2
                and np.ndim(points[0]) == 1
                and np.ndim(points[1]) == 1
                and not (
                    len(points[0]) == 2
                    and len(points[1]) == 2
                    and np.isscalar(points[0][0])
                    and not isinstance(points, tuple)
                )
            ):
                # Distinguish (x_1d, y_1d) tuple of 1D axes from an (N, 2) list
                x_1d = np.asarray(points[0], dtype=float)
                y_1d = np.asarray(points[1], dtype=float)
                X, Y = np.meshgrid(x_1d, y_1d, indexing="ij")
                self.points = np.column_stack([X.ravel(), Y.ravel()])
                self.out_shape = (x_1d.size, y_1d.size)
                self.grid_coords = [x_1d, y_1d]
            else:
                pts = np.atleast_2d(np.asarray(points, dtype=float))
                if pts.shape[1] != 2:
                    raise ValueError("`points` must have shape (n_pts, 2).")
                self.points = pts
                self.out_shape = (len(pts),)
                self.grid_coords = [pts[:, 0], pts[:, 1]]
        else:
            c = np.asarray(center, dtype=float)[:2]
            s = np.asarray(size, dtype=float)[:2]
            if np.all(s == 0.0):
                self.points = c[None, :]
                self.out_shape = (1,)
                self.grid_coords = [c[:1], c[1:2]]
            elif s[0] == 0.0:
                y = np.linspace(c[1] - 0.5 * s[1], c[1] + 0.5 * s[1], int(n_pts))
                x = np.full_like(y, c[0])
                self.points = np.column_stack([x, y])
                self.out_shape = (y.size,)
                self.grid_coords = [np.array([c[0]]), y]
            elif s[1] == 0.0:
                x = np.linspace(c[0] - 0.5 * s[0], c[0] + 0.5 * s[0], int(n_pts))
                y = np.full_like(x, c[1])
                self.points = np.column_stack([x, y])
                self.out_shape = (x.size,)
                self.grid_coords = [x, np.array([c[1]])]
            else:
                raise ValueError(
                    "Provide `points=(x_1d, y_1d)` for a 2D grid or one zero in `size`."
                )

        self.is_first_update = True

    def _initialize(self, solver):
        self.solver = solver
        self._freqs = self.frequencies
        n_f = self._freqs.size

        need_edge = any(c in ("Ex", "Ey", "Hx", "Hy") for c in self.components)
        need_node = any(c in ("Hz", "Ez", "u") for c in self.components)

        if need_edge:
            Mx, My = solver.edge_field_matrices(self.points)
            cols = np.unique(np.concatenate([Mx.indices, My.indices]))
            self._edge_cols = cols
            remap = np.zeros(solver.n_edges, dtype=int)
            remap[cols] = np.arange(cols.size)
            self._Mx = sp.csr_matrix(
                (Mx.data, remap[Mx.indices], Mx.indptr),
                shape=(len(self.points), cols.size),
            )
            self._My = sp.csr_matrix(
                (My.data, remap[My.indices], My.indptr),
                shape=(len(self.points), cols.size),
            )
            self._g_acc = np.zeros((n_f, cols.size), dtype=np.complex128)
        else:
            self._edge_cols = None
            self._g_acc = None

        if need_node:
            S_u = solver.interpolation(self.points)
            self._S_u, self._dof_cols = _restrict_sparse_columns(S_u)
            self._u_acc = np.zeros(
                (n_f, self._dof_cols.size), dtype=np.complex128
            )
        else:
            self._dof_cols = None
            self._u_acc = None

    def update(self, solver, t=None):
        """
        Accumulate fields after the step that started at time `t` [s].
        """
        if self.is_first_update:
            self._initialize(solver)
            self.is_first_update = False

        dt = solver.dt
        if t is None:
            t = (solver.n - 1) * dt
        phase = -2j * np.pi * self._freqs

        if self._g_acc is not None and self._edge_cols.size > 0:
            kg = dt * np.exp(phase * (t + 0.5 * dt))[:, None]
            self._g_acc += kg * solver.g[self._edge_cols][None, :]

        if self._u_acc is not None and self._dof_cols.size > 0:
            ku = dt * np.exp(phase * (t + dt))[:, None]
            self._u_acc += ku * solver.u[self._dof_cols][None, :]

    def get_field(self, component):
        """
        Return `(field, coords)` for `component`, with `field` of shape
        `(n_freq, *out_shape)`.
        """
        if component not in self.components:
            raise ValueError(
                f"Component {component!r} was not requested in {self.components}."
            )
        pol = self.solver.pol
        n_f = self.frequencies.size
        if component in ("Hz", "Ez", "u"):
            vals = (self._S_u @ self._u_acc.T).T
        elif pol == "TE":
            if component == "Ex":
                vals = (self._Mx @ self._g_acc.T).T
            elif component == "Ey":
                vals = (self._My @ self._g_acc.T).T
            else:
                raise ValueError(f"Component {component!r} unavailable in TE.")
        else:
            if component == "Hx":
                vals = -(self._Mx @ self._g_acc.T).T
            elif component == "Hy":
                vals = -(self._My @ self._g_acc.T).T
            else:
                raise ValueError(f"Component {component!r} unavailable in TM.")
        return vals.reshape((n_f, *self.out_shape)), self.grid_coords


class SubgridModeMonitor:
    """
    Forward and backward waveguide mode amplitudes `(a+, a-)` on a port line
    of `SubgridFIT2D`.

    Projects the Fourier-transformed nodal field `u` (at `t + dt`) and the
    two-sided average of the normal dual-edge voltages `g_a` (at `t + dt/2`)
    onto the unit-power `SubgridMode`s of `solve_port_modes_2d`:

    .. math::

        a^+ = \\frac{1}{4}(E \\times h^* + e^* \\times H), \\qquad
        a^- = \\frac{1}{4}(E \\times h^* - e^* \\times H).

    Parameters
    ----------
    frequencies : array_like
        Frequencies to monitor [Hz].
    center, size : sequence of float
        Port line centre and extent [m].
    n_modes : int, optional
        Number of modes to decompose into. Default 1.
    mode_frequency : float, optional
        If given, use the eigenmodes solved at `mode_frequency` [Hz] across the
        band (matching `PhotonicCircuit.run_s_parameters`). If None, solve the
        eigenmodes at every monitored frequency.
    """

    def __init__(self, frequencies, center, size, n_modes=1, mode_frequency=None):
        self.frequencies = np.atleast_1d(np.asarray(frequencies, dtype=float))
        self.center = np.asarray(center, dtype=float)
        self.size = np.asarray(size, dtype=float)
        self.n_modes = int(n_modes)
        self.mode_frequency = (
            None if mode_frequency is None else float(mode_frequency)
        )
        self.modes = None
        self.is_first_update = True

    def _initialize(self, solver):
        self.solver = solver
        self.line = line = _PortLine2D(solver, self.center, self.size)
        self._freqs = self.frequencies
        n_f = self._freqs.size
        M1 = line.n_seg + 1

        self._dofs = np.maximum(line.dofs, 0)
        self._dof_mask = (line.dofs >= 0).astype(float)

        # Sparse map (M + 1, n_edges) summing 0.5 * (g_lo + g_hi) onto each
        # line node (handles both regular 1-edge nodes and 2-edge transition
        # nodes on coarse/fine interfaces)
        rows, cols, vals = [], [], []
        for j in range(M1):
            for e_idx in line.e_lo_groups[j]:
                rows.append(j)
                cols.append(e_idx)
                vals.append(0.5)
            for e_idx in line.e_hi_groups[j]:
                rows.append(j)
                cols.append(e_idx)
                vals.append(0.5)
        G_map = sp.csr_matrix((vals, (rows, cols)), shape=(M1, solver.n_edges))
        self._G_sub, self._edge_cols = _restrict_sparse_columns(G_map)

        self._u_acc = np.zeros((n_f, M1), dtype=np.complex128)
        self._g_edge_acc = np.zeros(
            (n_f, self._edge_cols.size), dtype=np.complex128
        )

    def update(self, solver, t=None):
        """Accumulate port-line fields after the step that started at `t` [s]."""
        if self.is_first_update:
            self._initialize(solver)
            self.is_first_update = False

        dt = solver.dt
        if t is None:
            t = (solver.n - 1) * dt
        phase = -2j * np.pi * self._freqs

        ku = dt * np.exp(phase * (t + dt))[:, None]
        kg = dt * np.exp(phase * (t + 0.5 * dt))[:, None]

        u_vals = solver.u[self._dofs] * self._dof_mask
        self._u_acc += ku * u_vals[None, :]
        self._g_edge_acc += kg * solver.g[self._edge_cols][None, :]

    def _solve_modes(self):
        if self.mode_frequency is not None:
            modes = solve_port_modes_2d(
                self.solver,
                self.mode_frequency,
                self.center,
                self.size,
                n_modes=self.n_modes,
            )
            self.modes = [modes] * self.frequencies.size
        else:
            self.modes = [
                solve_port_modes_2d(
                    self.solver,
                    float(f),
                    self.center,
                    self.size,
                    n_modes=self.n_modes,
                )
                for f in self.frequencies
            ]

    def get_amplitudes(self, mode_index=0):
        """
        Return `(a_plus, a_minus)` at each monitored frequency [sqrt(W/m) s].
        """
        if self.modes is None:
            self._solve_modes()
        g_a_acc = (self._G_sub @ self._g_edge_acc.T).T  # (n_f, M + 1)
        u_acc = self._u_acc  # (n_f, M + 1)

        a_plus = np.full(self.frequencies.size, np.nan, dtype=complex)
        a_minus = np.full(self.frequencies.size, np.nan, dtype=complex)
        pol = self.solver.pol

        for i, modes in enumerate(self.modes):
            if len(modes) <= mode_index:
                continue
            m = modes[mode_index]
            if pol == "TE":
                # G_a = db^* E_t (with sign), U = H_z
                E_x_h = np.sum(g_a_acc[i] * np.conj(m.u))
                e_x_H = np.sum(np.conj(m.g_a_mean) * u_acc[i])
            else:
                # U = E_z, G_a = db^* H_t (with sign)
                E_x_h = np.sum(u_acc[i] * np.conj(m.g_a_mean))
                e_x_H = np.sum(np.conj(m.u) * g_a_acc[i])
            a_plus[i] = 0.25 * (E_x_h + e_x_H)
            a_minus[i] = 0.25 * (E_x_h - e_x_H)
        return a_plus, a_minus


class SubgridFluxMonitor:
    """
    Spectrum of the Poynting flux `Phi(f) = Re int (E(f) x H^*(f)) . n dl`
    through a line or out of a closed 2D box in `SubgridFIT2D`.

    For a single forward mode of amplitude `a+(f)`, `Phi(f) = 2 |a+(f)|^2`,
    matching `wakis.field_monitors.FluxMonitor`.

    Parameters
    ----------
    frequencies : array_like
        Frequencies to monitor [Hz].
    center : sequence of float, optional
        Centre `(x, y)` (or `(x, y, z)`) of the line or box [m].
    size : sequence of float, optional
        Extent `(sx, sy)` (or `(sx, sy, sz)`) [m]. One zero entry in `(sx, sy)`
        defines a line normal to that axis (`+axis` sign); two positive entries
        define a closed 2D box (`+outward` sign).
    segment : tuple (p0, p1), optional
        Alternative specification of a directed segment from `p0 = (x0, y0)` to
        `p1 = (x1, y1)` [m] with normal `n = (t_y, -t_x)`.
    n_pts : int, optional
        Number of quadrature sub-intervals along each segment. If None, chosen
        from the minimum cell size of the mesh (~2 samples per finest cell).
    """

    def __init__(
        self,
        frequencies,
        center=(0.0, 0.0),
        size=(0.0, 0.0),
        segment=None,
        n_pts=None,
    ):
        self.frequencies = np.atleast_1d(np.asarray(frequencies, dtype=float))
        self.center = np.asarray(center, dtype=float)[:2]
        self.size = np.asarray(size, dtype=float)[:2]
        self.segment = segment
        self.n_pts = n_pts
        self.is_first_update = True

    def _build_segments(self, solver):
        """Return list of `(p0, p1, normal_vec)` segments [m]."""
        m = solver.mesh
        xmin, xmax = float(m.xc[0]), float(m.xc[-1])
        ymin, ymax = float(m.yc[0]), float(m.yc[-1])

        if self.segment is not None:
            p0 = np.asarray(self.segment[0], dtype=float)[:2]
            p1 = np.asarray(self.segment[1], dtype=float)[:2]
            dp = p1 - p0
            L = float(np.linalg.norm(dp))
            n_vec = np.array([dp[1] / L, -dp[0] / L])
            return [(p0, p1, n_vec)]

        cx, cy = self.center
        sx, sy = self.size
        if sx == 0.0 and sy > 0.0:
            y0 = max(cy - 0.5 * sy, ymin) if np.isfinite(sy) else ymin
            y1 = min(cy + 0.5 * sy, ymax) if np.isfinite(sy) else ymax
            return [
                (np.array([cx, y0]), np.array([cx, y1]), np.array([1.0, 0.0]))
            ]
        if sy == 0.0 and sx > 0.0:
            x0 = max(cx - 0.5 * sx, xmin) if np.isfinite(sx) else xmin
            x1 = min(cx + 0.5 * sx, xmax) if np.isfinite(sx) else xmax
            return [
                (np.array([x0, cy]), np.array([x1, cy]), np.array([0.0, 1.0]))
            ]
        if sx > 0.0 and sy > 0.0:
            x0, x1 = cx - 0.5 * sx, cx + 0.5 * sx
            y0, y1 = cy - 0.5 * sy, cy + 0.5 * sy
            return [
                (np.array([x0, y0]), np.array([x0, y1]), np.array([-1.0, 0.0])),
                (np.array([x1, y0]), np.array([x1, y1]), np.array([1.0, 0.0])),
                (np.array([x0, y0]), np.array([x1, y0]), np.array([0.0, -1.0])),
                (np.array([x0, y1]), np.array([x1, y1]), np.array([0.0, 1.0])),
            ]
        raise ValueError("FluxMonitor requires a line or box with non-zero length.")

    def _initialize(self, solver):
        self.solver = solver
        self._freqs = self.frequencies
        n_f = self._freqs.size
        c = solver.mesh.cells
        h_min = float(
            np.min(np.minimum(c["x1"] - c["x0"], c["y1"] - c["y0"]))
        )

        pts_list, n_list, dl_list = [], [], []
        for p0, p1, n_vec in self._build_segments(solver):
            L = float(np.linalg.norm(p1 - p0))
            n_sub = (
                int(self.n_pts)
                if self.n_pts is not None
                else max(16, int(np.ceil(2.0 * L / h_min)))
            )
            t = (np.arange(n_sub, dtype=float) + 0.5) / n_sub
            pts = p0[None, :] + t[:, None] * (p1 - p0)[None, :]
            pts_list.append(pts)
            n_list.append(np.broadcast_to(n_vec, pts.shape))
            dl_list.append(np.full(n_sub, L / n_sub))

        pts_all = np.vstack(pts_list)
        n_all = np.vstack(n_list)
        dl_all = np.concatenate(dl_list)

        Mx, My = solver.edge_field_matrices(pts_all)
        S_u = solver.interpolation(pts_all)

        # Tangential electric/magnetic operator:
        # In TE: (E x H^*) . n = (n_x E_y - n_y E_x) H_z^*
        # In TM: (E x H^*) . n = E_z (n_y H_x^* - n_x H_y^*)
        #                      = E_z (-n_y Mx + n_x My) g^*
        w_edge = sp.diags(n_all[:, 0]) @ My - sp.diags(n_all[:, 1]) @ Mx
        self._M_edge, self._edge_cols = _restrict_sparse_columns(w_edge)
        self._S_u, self._dof_cols = _restrict_sparse_columns(S_u)
        self._pts = pts_all
        self._normals = n_all
        self._dl = dl_all

        self._g_acc = np.zeros((n_f, self._edge_cols.size), dtype=np.complex128)
        self._u_acc = np.zeros((n_f, self._dof_cols.size), dtype=np.complex128)

    def update(self, solver, t=None):
        """Accumulate fields after the step that started at `t` [s]."""
        if self.is_first_update:
            self._initialize(solver)
            self.is_first_update = False

        dt = solver.dt
        if t is None:
            t = (solver.n - 1) * dt
        phase = -2j * np.pi * self._freqs
        kg = dt * np.exp(phase * (t + 0.5 * dt))[:, None]
        ku = dt * np.exp(phase * (t + dt))[:, None]

        if self._edge_cols.size > 0:
            self._g_acc += kg * solver.g[self._edge_cols][None, :]
        if self._dof_cols.size > 0:
            self._u_acc += ku * solver.u[self._dof_cols][None, :]

    def get_flux(self):
        """Return the Poynting flux spectrum `Phi(f)` [W/m s^2]."""
        edge_f = (self._M_edge @ self._g_acc.T).T  # (n_f, n_pts)
        node_f = (self._S_u @ self._u_acc.T).T  # (n_f, n_pts)
        if self.solver.pol == "TE":
            density = np.real(edge_f * np.conj(node_f))
        else:
            density = np.real(node_f * np.conj(edge_f))
        return density @ self._dl


class SubgridNear2FarMonitor(SubgridFluxMonitor):
    """
    2D near-to-far-field transformation from a closed contour (or 3-sided box
    above a PEC ground plane) in `SubgridFIT2D`.

    Evaluates the 2D cylindrical asymptotics (`exp(-i k r) / sqrt(r)` in the
    `exp(+i w t)` convention) of the Franz / Green contour integral:

    * TM (`u = Ez`):
      ``F_TM(phi) = sqrt(k / (8 pi)) exp(i pi / 4) oint (eta * edge_f + (n' . r_hat) Ez) exp(i k r_hat . (r' - r0)) dl'``,
      with radiation intensity ``U(phi) = |F_TM(phi)|^2 / eta`` [W/(m rad) s^2];
    * TE (`u = Hz`):
      ``F_TE(phi) = sqrt(k / (8 pi)) exp(i pi / 4) oint (edge_f / eta + (n' . r_hat) Hz) exp(i k r_hat . (r' - r0)) dl'``,
      with ``E_phi = -eta * F_TE`` and radiation intensity
      ``U(phi) = eta * |F_TE(phi)|^2`` [W/(m rad) s^2].

    By Parseval's theorem, ``int_0^{2 pi} U(phi) dphi`` equals the outward
    Poynting flux `get_flux()`.

    Parameters
    ----------
    frequencies : array_like
        Frequencies to monitor [Hz].
    center : sequence of float, optional
        Box or line centre `(x, y)` [m] (also the phase reference `r0` unless
        `origin` is provided).
    size : sequence of float, optional
        Box or line extent `(sx, sy)` [m].
    origin : sequence of float, optional
        Phase origin `r0 = (x0, y0)` [m] for the far-field pattern. Defaults
        to `center`.
    ground_plane : {'-y', '+y', '-x', '+x'} or None, optional
        If set, omits the corresponding box wall and includes the exact image
        contour across that PEC ground plane.
    n_pts : int, optional
        Quadrature points per box side.
    """

    def __init__(
        self,
        frequencies,
        center=(0.0, 0.0),
        size=(0.0, 0.0),
        origin=None,
        ground_plane=None,
        n_pts=None,
    ):
        super().__init__(
            frequencies=frequencies,
            center=center,
            size=size,
            n_pts=n_pts,
        )
        self.origin = (
            self.center.copy()
            if origin is None
            else np.asarray(origin, dtype=float)[:2]
        )
        if ground_plane is not None and ground_plane not in ("-x", "+x", "-y", "+y"):
            raise ValueError(
                "ground_plane must be one of '-x', '+x', '-y', '+y', or None."
            )
        self.ground_plane = ground_plane

    def _initialize(self, solver):
        super()._initialize(solver)
        c = solver.mesh.cells
        h_min = float(np.min(np.minimum(c["x1"] - c["x0"], c["y1"] - c["y0"])))
        eps_n = 1e-6 * h_min
        xmin, xmax = float(solver.mesh.xc[0]), float(solver.mesh.xc[-1])
        ymin, ymax = float(solver.mesh.yc[0]), float(solver.mesh.yc[-1])
        pts_p = np.column_stack(
            [
                np.clip(self._pts[:, 0] + eps_n * self._normals[:, 0], xmin + 1e-12, xmax - 1e-12),
                np.clip(self._pts[:, 1] + eps_n * self._normals[:, 1], ymin + 1e-12, ymax - 1e-12),
            ]
        )
        pts_m = np.column_stack(
            [
                np.clip(self._pts[:, 0] - eps_n * self._normals[:, 0], xmin + 1e-12, xmax - 1e-12),
                np.clip(self._pts[:, 1] - eps_n * self._normals[:, 1], ymin + 1e-12, ymax - 1e-12),
            ]
        )
        Mx_p, My_p = solver.edge_field_matrices(pts_p)
        Mx_m, My_m = solver.edge_field_matrices(pts_m)
        Mx = 0.5 * (Mx_p + Mx_m)
        My = 0.5 * (My_p + My_m)
        w_edge = sp.diags(self._normals[:, 0]) @ My - sp.diags(self._normals[:, 1]) @ Mx
        self._M_edge, self._edge_cols = _restrict_sparse_columns(w_edge)
        self._g_acc = np.zeros((self._freqs.size, self._edge_cols.size), dtype=np.complex128)

    def _build_segments(self, solver):
        segs = super()._build_segments(solver)
        if self.ground_plane is None or len(segs) != 4:
            return segs
        # Order in SubgridFluxMonitor._build_segments for a box:
        # 0: -x, 1: +x, 2: -y, 3: +y
        drop = {"-x": 0, "+x": 1, "-y": 2, "+y": 3}[self.ground_plane]
        return [s for i, s in enumerate(segs) if i != drop]

    def far_field(self, phi, eps_r=None):
        """
        Compute 2D far-field pattern at angles `phi` [rad] measured from `+x`.

        Parameters
        ----------
        phi : array_like
            Observation angles [rad], shape `(n_phi,)`.
        eps_r : float, optional
            Background relative permittivity. Defaults to `solver.geo.background_eps`.

        Returns
        -------
        dict of ndarray
            * `'Ez'`, `'H_phi'` in TM or `'Hz'`, `'E_phi'` in TE, shape `(n_freq, n_phi)`;
            * `'intensity'`: radiation intensity `U(phi)` [W/(m rad) s^2], shape `(n_freq, n_phi)`;
            * `'phi'`: angles `phi` [rad], shape `(n_phi,)`.
        """
        phi_arr = np.atleast_1d(np.asarray(phi, dtype=float))
        eb = float(self.solver.geo.background_eps if eps_r is None else eps_r)
        nb = np.sqrt(eb)
        eta = np.sqrt(MU0 / EPS0) / nb
        k = (2.0 * np.pi * self.frequencies * nb / C0)[:, None, None]  # (n_f, 1, 1)

        r_hat = np.stack([np.cos(phi_arr), np.sin(phi_arr)], axis=-1)  # (n_phi, 2)
        edge_f = (self._M_edge @ self._g_acc.T).T  # (n_f, n_pts)
        node_f = (self._S_u @ self._u_acc.T).T  # (n_f, n_pts)

        pts = self._pts
        normals = self._normals
        dl = self._dl
        pol = self.solver.pol

        def _contrib(pts_c, normals_c, node_c, edge_c):
            dr = pts_c - self.origin[None, :]  # (n_pts, 2)
            proj = r_hat @ dr.T  # (n_phi, n_pts)
            n_dot_r = (r_hat @ normals_c.T)[None, :, :]  # (1, n_phi, n_pts)
            phase = np.exp(1j * k * proj[None, :, :])  # (n_f, n_phi, n_pts)
            if pol == "TM":
                integrand = eta * edge_c[:, None, :] + n_dot_r * node_c[:, None, :]
            else:
                integrand = (edge_c[:, None, :] / eta) + n_dot_r * node_c[:, None, :]
            return np.sum(integrand * phase * dl[None, None, :], axis=-1)

        integral = _contrib(pts, normals, node_f, edge_f)

        illum_mask = np.ones(phi_arr.shape, dtype=float)
        if self.ground_plane is not None:
            cx, cy = self.center
            sx, sy = self.size
            pts_img = pts.copy()
            normals_img = normals.copy()
            if self.ground_plane == "-y":
                y0 = cy - 0.5 * sy
                pts_img[:, 1] = 2.0 * y0 - pts[:, 1]
                normals_img[:, 1] = -normals[:, 1]
                illum_mask = (np.sin(phi_arr) >= -1e-12).astype(float)
            elif self.ground_plane == "+y":
                y0 = cy + 0.5 * sy
                pts_img[:, 1] = 2.0 * y0 - pts[:, 1]
                normals_img[:, 1] = -normals[:, 1]
                illum_mask = (np.sin(phi_arr) <= 1e-12).astype(float)
            elif self.ground_plane == "-x":
                x0 = cx - 0.5 * sx
                pts_img[:, 0] = 2.0 * x0 - pts[:, 0]
                normals_img[:, 0] = -normals[:, 0]
                illum_mask = (np.cos(phi_arr) >= -1e-12).astype(float)
            elif self.ground_plane == "+x":
                x0 = cx + 0.5 * sx
                pts_img[:, 0] = 2.0 * x0 - pts[:, 0]
                normals_img[:, 0] = -normals[:, 0]
                illum_mask = (np.cos(phi_arr) <= 1e-12).astype(float)
            img_sign = -1.0 if pol == "TM" else 1.0
            integral = integral + img_sign * _contrib(
                pts_img, normals_img, node_f, edge_f
            )

        prefac = np.sqrt(k[:, :, 0] / (8.0 * np.pi)) * np.exp(0.25j * np.pi)
        F = prefac * integral * illum_mask[None, :]
        if pol == "TM":
            Ez = F
            H_phi = -F / eta
            intensity = (np.abs(F) ** 2) / eta
            return dict(Ez=Ez, H_phi=H_phi, intensity=intensity, phi=phi_arr)
        else:
            Hz = F
            E_phi = -eta * F
            intensity = eta * (np.abs(F) ** 2)
            return dict(Hz=Hz, E_phi=E_phi, intensity=intensity, phi=phi_arr)

    def radiated_power(self, n_phi=360, eps_r=None):
        """Total far-field radiated power `int_0^{2 pi} U(phi) dphi` [W/m s^2]."""
        phi = np.linspace(0.0, 2.0 * np.pi, int(n_phi), endpoint=False)
        dphi = 2.0 * np.pi / int(n_phi)
        ff = self.far_field(phi, eps_r=eps_r)
        return np.sum(ff["intensity"], axis=-1) * dphi

    def directivity(self, phi, n_phi=360, eps_r=None):
        """2D angular directivity `2 pi U(phi) / P_rad` [dimensionless]."""
        ff = self.far_field(phi, eps_r=eps_r)
        P_rad = self.radiated_power(n_phi=n_phi, eps_r=eps_r)
        return (2.0 * np.pi * ff["intensity"]) / np.maximum(P_rad[:, None], 1e-300)


Near2FarMonitor2D = SubgridNear2FarMonitor


class SubgridDiffractionMonitor(SubgridFluxMonitor):
    """
    Floquet diffraction-order monitor on a line spanning a periodic/Bloch unit
    cell (`mesh.periodic_y == True`) in `SubgridFIT2D`.

    Decomposes the tangential fields on a line normal to `x` (`size=(0, Ly)`)
    into Floquet orders `m` with transverse wavevector

    .. math::

        k_{y,m} = k_{y,\\mathrm{inc}} + \\frac{2\\pi m}{L_y},

    where ``bloch_phase = exp(-i k_{y,inc} L_y)``. By Parseval's theorem, the
    sum of `net` over all orders equals `get_flux()` to machine precision, and
    for every propagating order ``up - down == net``.

    Parameters
    ----------
    frequencies : array_like
        Monitored frequencies [Hz] (must be non-zero).
    center : sequence of float, optional
        Point `(x0, y0)` on the monitor line [m].
    size : sequence of float, optional
        Line extent `(0.0, Ly)` [m] (`sx == 0`). Automatically spans the full
        unit-cell period `[ymin, ymax]`.
    n_pts : int, optional
        Number of uniform Fourier sample points across `Ly`.
    """

    def __init__(self, frequencies, center=(0.0, 0.0), size=(0.0, np.inf), n_pts=None):
        freqs = np.atleast_1d(np.asarray(frequencies, dtype=float))
        if np.any(freqs == 0.0):
            raise ValueError("frequencies must be nonzero.")
        s = np.asarray(size, dtype=float)[:2]
        if s[0] != 0.0:
            raise ValueError(
                "SubgridDiffractionMonitor requires an x-normal line (size[0] == 0)."
            )
        super().__init__(
            frequencies=freqs,
            center=center,
            size=(0.0, np.inf),
            n_pts=n_pts,
        )

    def _initialize(self, solver):
        if not solver.mesh.periodic_y:
            raise ValueError(
                "SubgridDiffractionMonitor requires `mesh.periodic_y == True`."
            )
        super()._initialize(solver)
        m = solver.mesh
        self.Ly = float(m.yc[-1] - m.yc[0])
        self.y0 = float(m.yc[0])
        self.n_orders = self._pts.shape[0]
        self.ky_inc = - float(np.angle(solver.bloch_phase)) / self.Ly
        m_int = np.fft.fftfreq(self.n_orders) * self.n_orders
        self._orders = m_int.astype(int)
        self.ky = self.ky_inc + 2.0 * np.pi * self._orders / self.Ly
        eb = float(solver.geo.background_eps)
        self.k0 = 2.0 * np.pi * self.frequencies * np.sqrt(eb) / C0
        kx2 = self.k0[:, None] ** 2 - self.ky[None, :] ** 2
        self._propagating = kx2 > 0.0
        self.kx = np.sqrt(np.maximum(kx2, 0.0))

    @property
    def orders(self):
        """1D integer array of Floquet order indices `m`."""
        return self._orders

    @property
    def propagating(self):
        """Boolean mask `(n_freq, n_orders)` of propagating Floquet orders."""
        return self._propagating

    def _decompose(self):
        edge_f = (self._M_edge @ self._g_acc.T).T  # (n_f, N)
        node_f = (self._S_u @ self._u_acc.T).T  # (n_f, N)
        yj = self._pts[:, 1]
        dy = self.Ly / self.n_orders
        # Project onto exp(-i k_{y,m} y): U_m = (1/Ly) sum_j node_f(y_j) exp(+i k_{y,m} y_j) dy
        basis = np.exp(1j * yj[:, None] * self.ky[None, :]) * (dy / self.Ly)
        U = node_f @ basis  # (n_f, n_orders)
        V = edge_f @ basis  # (n_f, n_orders)
        net = self.Ly * np.real(U * np.conj(V))

        eb = float(self.solver.geo.background_eps)
        eta_b = np.sqrt(MU0 / EPS0) / np.sqrt(eb)
        cos_theta = np.where(
            self._propagating,
            self.kx / np.maximum(self.k0[:, None], 1e-300),
            0.0,
        )
        # Wave admittance (TM: V = Y U) or wave impedance (TE: V = Z U)
        if self.solver.pol == "TM":
            W_imp = cos_theta / eta_b
        else:
            W_imp = eta_b * cos_theta

        sqrt_W = np.sqrt(np.maximum(W_imp, 0.0))
        inv_sqrt_W = np.where(
            self._propagating,
            1.0 / np.where(self._propagating, sqrt_W, 1.0),
            0.0,
        )
        scale = 0.5 * np.sqrt(self.Ly)
        a_up = np.where(
            self._propagating, scale * (sqrt_W * U + inv_sqrt_W * V), 0.0
        )
        a_down = np.where(
            self._propagating, scale * (sqrt_W * U - inv_sqrt_W * V), 0.0
        )
        return a_up, a_down, net

    def get_amplitudes(self):
        """
        Return complex amplitudes of propagating Floquet orders [sqrt(W/m) s].

        Returns
        -------
        dict of ndarray
            Keys `'up'`, `'down'`, plus `'up_s'`, `'down_s'` (TM) and
            `'up_p'`, `'down_p'` (TE), each of shape `(n_freq, n_orders)`.
        """
        a_up, a_down, _ = self._decompose()
        z = np.zeros_like(a_up)
        if self.solver.pol == "TM":
            return dict(
                up=a_up, down=a_down, up_s=a_up, down_s=a_down, up_p=z, down_p=z
            )
        return dict(
            up=a_up, down=a_down, up_s=z, down_s=z, up_p=a_up, down_p=a_down
        )

    def get_power(self, incident=None):
        """
        Return power spectra `'up'`, `'down'`, `'net'` per Floquet order
        (shape `(n_freq, n_orders)`), optionally normalized by `incident`.
        """
        a_up, a_down, net = self._decompose()
        out = dict(
            up=np.abs(a_up) ** 2,
            down=np.abs(a_down) ** 2,
            net=net,
        )
        if incident is not None:
            inc = np.atleast_1d(np.asarray(incident, dtype=float))[:, None]
            for k in out:
                out[k] = out[k] / inc
        return out

    def order_angles(self, degrees=False):
        """
        Propagation angles `theta_m = arcsin(k_{y,m} / k)` [rad or deg] of
        shape `(n_freq, n_orders)` (`nan` for evanescent orders).
        """
        ratio = self.ky[None, :] / np.maximum(self.k0[:, None], 1e-300)
        angles = np.where(
            self._propagating,
            np.arcsin(np.clip(ratio, -1.0, 1.0)),
            np.nan,
        )
        return np.degrees(angles) if degrees else angles

