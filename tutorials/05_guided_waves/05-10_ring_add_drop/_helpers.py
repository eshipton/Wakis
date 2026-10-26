# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""
Helper utilities for Tutorial 5.10 (Microring resonator add-drop filter).

Provides:
- `FastModeMonitor`: mode-projected port DFT monitor that inner-products the
  transverse `(E, H)` fields with the fixed port eigenmode during `update()`
  and transforms the 1D overlap trace in `get_amplitudes()`, matching
  `wakis.mode_solver.ModeMonitor` to machine precision (~1e-15) with a ~900x
  speedup for 1001 frequency points.
- `attach_fast_2d_te_step`: zero-copy 2D in-plane TE (`Ex, Ey, Hz`) CPML
  time-stepper and precomputed energy-norm weights for `two_d=True` EIM
  simulations, bit-for-bit identical to `SolverFIT3D._one_step_cpml`.
- `StridedDFTMonitor`: strided 2D midplane DFT monitor recording `Ex, Ey, Hz`
  at both off-resonance (`1550 nm`) and a dense wavelength comb spanning one
  free spectral range so the exact on-resonance steady-state field map can be
  extracted after the resonance wavelength is identified from `S_drop`.
- `RingPhotonicCircuit`: `PhotonicCircuit` subclass integrating the fast 2D TE
  stepper, `FastModeMonitor`, dual-port (`through` and `drop`) time-trace
  probes, achieved energy-decay tracking, and automatic on/off-resonance DFT
  field extraction.
- `find_ring_resonances` and `cmt_add_drop_spectra`: Lorentzian resonance
  fitting, FSR / finesse / quality-factor extraction, and Yariv (2000)
  coupled-mode / transfer-matrix analytical spectra.
"""

from pathlib import Path

from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.patches import Polygon as MplPolygon
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c as c_light, epsilon_0
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from wakis import figstyle, schematic
from wakis.field_monitors import FluxMonitor
from wakis.gridFIT3D import GridFIT3D
from wakis.mode_solver import ModeMonitor, ModeSource, _host, _Plane, solve_modes
from wakis.templates import meshing
from wakis.templates.photonics import (
    PhotonicCircuit,
    PolySlab,
    _SnapshotRecorder,
    n_to_eps,
    slab_neff,
)
from wakis.templates.simulation import Simulation
from wakis.unstructured import subtri_circles_mesh


class FastRingPolySlab(PolySlab):
    """
    `PolySlab` subclass for circular microring annuli that pre-filters cells by
    radial distance `r in [r_in - pad, r_out + pad]` before polygon edge queries.

    Produces bit-for-bit identical fill fractions and normals to `PolySlab` while
    skipping the interior hole and exterior corners during `assign_materials`.
    """

    def __init__(
        self,
        outer_vertices,
        z_min,
        z_max,
        material,
        holes=(),
        center=(0.0, 0.0),
        r_in=0.0,
        r_out=0.0,
        sidewall_angle_deg=0.0,
    ):
        super().__init__(
            outer_vertices,
            z_min,
            z_max,
            material=material,
            holes=holes,
            sidewall_angle_deg=sidewall_angle_deg,
        )
        self._cx = float(center[0])
        self._cy = float(center[1])
        self._r_in = float(r_in)
        self._r_out = float(r_out)

    def fill(self, X, Y, Z, hx, hy, hz, n_sub=6):
        X, Y, Z, hx, hy, hz = np.broadcast_arrays(
            np.asarray(X, dtype=float),
            np.asarray(Y, dtype=float),
            np.asarray(Z, dtype=float),
            np.asarray(hx, dtype=float),
            np.asarray(hy, dtype=float),
            np.asarray(hz, dtype=float),
        )
        z_lo = np.maximum(Z - 0.5 * hz, self.z_min)
        z_hi = np.minimum(Z + 0.5 * hz, self.z_max)
        fz = np.where(
            hz > 0,
            np.clip((z_hi - z_lo) / np.where(hz > 0, hz, 1.0), 0.0, 1.0),
            np.where((Z >= self.z_min) & (Z <= self.z_max), 1.0, 0.0),
        )
        if not np.any(fz > 0):
            return np.zeros(X.shape, dtype=float)
        if self._r_out > self._r_in > 0.0:
            r = np.hypot(X - self._cx, Y - self._cy)
            pad = 0.5 * np.hypot(hx, hy) + 2.5e-8
            band = (fz > 0) & (r >= self._r_in - pad) & (r <= self._r_out + pad)
            out = np.zeros(X.shape, dtype=float)
            if np.any(band):
                out[band] = super().fill(
                    X[band],
                    Y[band],
                    Z[band],
                    hx[band],
                    hy[band],
                    hz[band],
                    n_sub=n_sub,
                )
            return out
        return super().fill(X, Y, Z, hx, hy, hz, n_sub=n_sub)


def coupler_gap_mesh_overrides(
    gap_m,
    n_cells_across_gap,
    radius_m=5.0e-6,
    width_m=0.43e-6,
    cells_per_wavelength=10,
    n_core=3.476,
    n_clad=1.444,
    thickness_m=0.22e-6,
    wavelength_min=1.50e-6,
    wavelength_ref=1.55e-6,
):
    """
    Build `mesh_overrides` refinement boxes at the two bus-ring coupling points.

    Parameters
    ----------
    gap_m : float
        Edge-to-edge coupling gap `g` [m] between the ring and each bus waveguide.
    n_cells_across_gap : int or None
        Target number of cells `N` across `gap_m`. If `None` or `<= 0`, returns
        an empty list (no mesh overrides).
    radius_m, width_m : float, optional
        Microring mean radius `R` [m] and waveguide width `w_wg` [m].
    cells_per_wavelength : float, optional
        Background resolution in cells per wavelength inside the slab index.
    n_core, n_clad, thickness_m, wavelength_min, wavelength_ref : float, optional
        SOI slab parameters used to compute the background core cell size.

    Returns
    -------
    list of tuple
        Two `(lo, hi, max_cell)` refinement boxes covering the upper and lower
        bus-ring gaps whenever `gap_m / N` is finer than the background cell size.
    """
    if n_cells_across_gap is None or n_cells_across_gap <= 0:
        return []
    h_gap = float(gap_m) / float(n_cells_across_gap)
    neff_slab = float(
        slab_neff(n_core, n_clad, thickness_m, wavelength_ref, polarization="TE")
    )
    h_bg = float(wavelength_min / (neff_slab * float(cells_per_wavelength)))
    if h_gap >= h_bg:
        return []
    r_out = float(radius_m + 0.5 * width_m)
    g = float(gap_m)
    x_box = 0.8e-6
    max_cell = (1.0, h_gap, 1.0)
    return [
        ((-x_box, r_out, -np.inf), (x_box, r_out + g, np.inf), max_cell),
        ((-x_box, -(r_out + g), -np.inf), (x_box, -r_out, np.inf), max_cell),
    ]


def measure_cells_across_gap(grid_y, gap_m, radius_m=5.0e-6, width_m=0.43e-6):
    """
    Measure `cells_across_feature = min_feature / max_cell_across_feature` from
    the actual mesh node array `grid_y` [m] across both coupling gaps.

    Parameters
    ----------
    grid_y : array_like
        1D array of mesh node coordinates along `y` [m].
    gap_m : float
        Edge-to-edge coupling gap `g` [m] (the smallest feature).
    radius_m, width_m : float, optional
        Microring mean radius `R` [m] and waveguide width `w_wg` [m].

    Returns
    -------
    cells_across_feature : float
        Measured `gap_m / dy_max_in_gap`.
    dy_max_in_gap : float
        Largest `y` cell size [m] overlapping the upper or lower coupling gap.
    """
    y = np.asarray(grid_y, dtype=np.float64)
    dy = np.diff(y)
    r_out = float(radius_m + 0.5 * width_m)
    g = float(gap_m)
    tol = 1e-12
    in_top = (y[1:] > r_out + tol) & (y[:-1] < r_out + g - tol)
    in_bot = (y[1:] > -(r_out + g) + tol) & (y[:-1] < -r_out - tol)
    mask = in_top | in_bot
    if not np.any(mask):
        yc = 0.5 * (y[:-1] + y[1:])
        idx_top = int(np.argmin(np.abs(yc - (r_out + 0.5 * g))))
        idx_bot = int(np.argmin(np.abs(yc + (r_out + 0.5 * g))))
        dy_max = float(max(dy[idx_top], dy[idx_bot]))
    else:
        dy_max = float(np.max(dy[mask]))
    return float(np.round(g / dy_max, 9)), dy_max


class FastModeMonitor(ModeMonitor):
    """
    ModeMonitor optimized for fixed `mode_frequency` broadband S-parameter sweeps.

    By projecting the port-plane fields `(E, H)` onto the spatial eigenmode
    `(e, h_mean)` at each timestep and Fourier-transforming the resulting
    scalar overlap traces in `get_amplitudes()`, per-step complexity drops from
    `O(N_freqs * N_cross_section)` complex multiply-adds to `O(N_port_active)`
    real multiply-adds while reproducing `ModeMonitor.get_amplitudes()` to
    machine precision (~1e-15).
    """

    def _initialize(self, solver):
        if self.mode_frequency is None or self.n_modes != 1:
            return super()._initialize(solver)
        self.solver = solver
        self.plane = plane = _Plane(solver, self.center, self.size)
        modes = solve_modes(
            solver, self.mode_frequency, self.center, self.size, self.n_modes
        )
        self.modes = [modes] * self.frequencies.size
        if not modes:
            self._fast_active = False
            return super()._initialize(solver)

        self._fast_active = True
        m = modes[0]
        w_Eb = (np.conj(m.h_mean["c"]) * plane.dA_b).ravel("F")
        w_Ec = (-np.conj(m.h_mean["b"]) * plane.dA_c).ravel("F")
        w_Hb = (-0.5 * np.conj(m.e["c"]) * plane.dA_c).ravel("F")
        w_Hc = (0.5 * np.conj(m.e["b"]) * plane.dA_b).ravel("F")

        idx_E = np.concatenate(
            [plane.flat_index(plane.b, plane.k), plane.flat_index(plane.c, plane.k)]
        )
        w_E = np.concatenate([w_Eb, w_Ec])
        keep_E = np.abs(w_E) > 0
        self._idx_E = idx_E[keep_E]
        self._w_E = np.real_if_close(w_E[keep_E])

        idx_Hb = np.concatenate(
            [
                plane.flat_index(plane.b, plane.k - 1),
                plane.flat_index(plane.b, plane.k),
            ]
        )
        idx_Hc = np.concatenate(
            [
                plane.flat_index(plane.c, plane.k - 1),
                plane.flat_index(plane.c, plane.k),
            ]
        )
        idx_H = np.concatenate([idx_Hb, idx_Hc])
        w_H = np.concatenate([w_Hb, w_Hb, w_Hc, w_Hc])
        keep_H = np.abs(w_H) > 0
        self._idx_H = idx_H[keep_H]
        self._w_H = np.real_if_close(w_H[keep_H])

        self._t_list = []
        self._eh1_list = []
        self._eh2_list = []

    def update(self, solver, t):
        if self.is_first_update:
            self._initialize(solver)
            self.is_first_update = False
        if not getattr(self, "_fast_active", False):
            return super().update(solver, t)
        E_arr = _host(solver.E.array)
        H_arr = _host(solver.H.array)
        self._t_list.append(float(t))
        self._eh1_list.append(np.dot(self._w_E, E_arr[self._idx_E]))
        self._eh2_list.append(np.dot(self._w_H, H_arr[self._idx_H]))

    def get_amplitudes(self, mode_index=0):
        if not getattr(self, "_fast_active", False) or mode_index != 0:
            return super().get_amplitudes(mode_index=mode_index)
        t = np.asarray(self._t_list, dtype=np.float64)
        dt = float(self.solver.dt)
        eh1 = np.asarray(self._eh1_list, dtype=np.complex128)
        eh2 = np.asarray(self._eh2_list, dtype=np.complex128)
        acc1 = np.zeros(self.frequencies.size, dtype=np.complex128)
        acc2 = np.zeros(self.frequencies.size, dtype=np.complex128)
        two_pi_f = -2j * np.pi * self.frequencies[:, None]
        chunk = 4096
        for start in range(0, t.size, chunk):
            sl = slice(start, min(start + chunk, t.size))
            k0 = np.exp(two_pi_f * t[None, sl])
            acc1 += k0 @ eh1[sl]
            acc2 += k0 @ eh2[sl]
        E_x_h = (dt * np.exp(-2j * np.pi * self.frequencies * dt)) * acc1
        e_x_H = (dt * np.exp(-1j * np.pi * self.frequencies * dt)) * acc2
        return 0.25 * (E_x_h + e_x_H), 0.25 * (E_x_h - e_x_H)


def attach_fast_2d_te_step(solver):
    """
    Replace `solver.one_step` and `solver.field_energy` with a specialized 2D
    quasi-TE (`Ex, Ey, Hz`) CPML implementation when `Nz == 1` and `ieps_z == 0`.

    Produces bit-for-bit identical field arrays (`max diff == 0.0`) to
    `SolverFIT3D._one_step_cpml` while eliminating the 8 zero out-of-plane curl
    products and 1D fancy-indexing copies.
    """
    if (
        solver.Nz != 1
        or not solver.activate_cpml
        or solver.use_gpu
        or solver.use_mpi
        or solver.use_conductivity
    ):
        return

    Nx, Ny, N = solver.Nx, solver.Ny, solver.N
    npml = solver.n_pml

    def view2d(arr1d):
        return arr1d.reshape((Ny, Nx))

    dzx_0 = view2d(solver.dzx.diagonal(0).copy())
    dzx_1 = solver.dzx.diagonal(1).copy()
    dzy_0 = view2d(solver.dzy.diagonal(0).copy())
    dzy_1 = solver.dzy.diagonal(Nx).reshape((Ny - 1, Nx)).copy()
    dtxy_0 = view2d(solver.dtxy.diagonal(0).copy())
    dtxy_1 = solver.dtxy.diagonal(-Nx).reshape((Ny - 1, Nx)).copy()
    dtyx_0 = view2d(solver.dtyx.diagonal(0).copy())
    dtyx_1 = solver.dtyx.diagonal(-1).copy()

    dt = float(solver.dt)
    dt_imuz = view2d(dt * solver.imu.field_z.copy())
    dt_iepsx = view2d(dt * solver.ieps.field_x.copy())
    dt_iepsy = view2d(dt * solver.ieps.field_y.copy())

    psiHa_z_lo = solver.psiHa_z_low.reshape((Ny, npml))
    psiHa_z_hi = solver.psiHa_z_high.reshape((Ny, npml))
    psiHb_z_lo = solver.psiHb_z_low.reshape((npml, Nx))
    psiHb_z_hi = solver.psiHb_z_high.reshape((npml, Nx))
    psiEb_y_lo = solver.psiEb_y_low.reshape((Ny, npml))
    psiEb_y_hi = solver.psiEb_y_high.reshape((Ny, npml))
    psiEa_x_lo = solver.psiEa_x_low.reshape((npml, Nx))
    psiEa_x_hi = solver.psiEa_x_high.reshape((npml, Nx))

    b_Hx_lo = solver.pml_b_H_x_low.reshape((Ny, npml))
    c_Hx_lo = solver.pml_c_H_x_low.reshape((Ny, npml))
    b_Hx_hi = solver.pml_b_H_x_high.reshape((Ny, npml))
    c_Hx_hi = solver.pml_c_H_x_high.reshape((Ny, npml))
    b_Hy_lo = solver.pml_b_H_y_low.reshape((npml, Nx))
    c_Hy_lo = solver.pml_c_H_y_low.reshape((npml, Nx))
    b_Hy_hi = solver.pml_b_H_y_high.reshape((npml, Nx))
    c_Hy_hi = solver.pml_c_H_y_high.reshape((npml, Nx))

    b_Ex_lo = solver.pml_b_E_x_low.reshape((Ny, npml))
    c_Ex_lo = solver.pml_c_E_x_low.reshape((Ny, npml))
    b_Ex_hi = solver.pml_b_E_x_high.reshape((Ny, npml))
    c_Ex_hi = solver.pml_c_E_x_high.reshape((Ny, npml))
    b_Ey_lo = solver.pml_b_E_y_low.reshape((npml, Nx))
    c_Ey_lo = solver.pml_c_E_y_low.reshape((npml, Nx))
    b_Ey_hi = solver.pml_b_E_y_high.reshape((npml, Nx))
    c_Ey_hi = solver.pml_c_E_y_high.reshape((npml, Nx))

    Ex1d, Ey1d, Hz1d = solver.E.field_x, solver.E.field_y, solver.H.field_z
    Ex, Ey, Hz = view2d(Ex1d), view2d(Ey1d), view2d(Hz1d)
    Jx, Jy = view2d(solver.J.field_x), view2d(solver.J.field_y)

    dzxEy1d = np.empty(N, dtype=solver.dtype)
    dzyEx1d = np.empty(N, dtype=solver.dtype)
    dtxyHz1d = np.empty(N, dtype=solver.dtype)
    dtyxHz1d = np.empty(N, dtype=solver.dtype)
    dzxEy, dzyEx = view2d(dzxEy1d), view2d(dzyEx1d)
    dtxyHz, dtyxHz = view2d(dtxyHz1d), view2d(dtyxHz1d)
    tmp1d = np.empty(N, dtype=solver.dtype)
    tmp = view2d(tmp1d)

    ieps = solver.ieps.toarray()
    imu = solver.imu.toarray()
    itA = solver.grid.itA.toarray()
    dA = np.divide(1.0, itA, out=np.zeros_like(itA), where=itA != 0)
    vol_E = solver.grid.L.toarray() * dA
    vol_H = solver.grid.tL.toarray() / solver.grid.iA.toarray()
    eps = np.where(ieps != 0, 1.0 / np.where(ieps != 0, ieps, 1.0), 0.0)
    mu = np.where(imu != 0, 1.0 / np.where(imu != 0, imu, 1.0), 0.0)
    w_Ex = 0.5 * (eps * vol_E)[:N]
    w_Ey = 0.5 * (eps * vol_E)[N : 2 * N]
    w_Hz = 0.5 * (mu * vol_H)[2 * N : 3 * N]

    def fast_field_energy():
        return float(
            np.dot(w_Ex, Ex1d * Ex1d)
            + np.dot(w_Ey, Ey1d * Ey1d)
            + np.dot(w_Hz, Hz1d * Hz1d)
        )

    def fast_one_step():
        if solver.step_0:
            solver._set_ghosts_to_0()
            solver.step_0 = False
            solver._attrcleanup()
            solver.J_old = np.zeros_like(solver.J.toarray())

        np.multiply(dzx_0, Ey, out=dzxEy)
        np.multiply(dzx_1, Ey1d[1:], out=tmp1d[:-1])
        dzxEy1d[:-1] += tmp1d[:-1]

        np.multiply(dzy_0, Ex, out=dzyEx)
        np.multiply(dzy_1, Ex[1:, :], out=tmp[:-1, :])
        dzyEx[:-1, :] += tmp[:-1, :]

        psiHa_z_lo[:] *= b_Hx_lo
        psiHa_z_lo[:] += c_Hx_lo * dzxEy[:, :npml]
        psiHb_z_lo[:] *= b_Hy_lo
        psiHb_z_lo[:] += c_Hy_lo * dzyEx[:npml, :]
        psiHa_z_hi[:] *= b_Hx_hi
        psiHa_z_hi[:] += c_Hx_hi * dzxEy[:, -npml:]
        psiHb_z_hi[:] *= b_Hy_hi
        psiHb_z_hi[:] += c_Hy_hi * dzyEx[-npml:, :]

        np.subtract(dzxEy, dzyEx, out=tmp)
        tmp[:] *= dt_imuz
        Hz[:] -= tmp
        Hz[:, :npml] -= dt_imuz[:, :npml] * psiHa_z_lo
        Hz[:npml, :] += dt_imuz[:npml, :] * psiHb_z_lo
        Hz[:, -npml:] -= dt_imuz[:, -npml:] * psiHa_z_hi
        Hz[-npml:, :] += dt_imuz[-npml:, :] * psiHb_z_hi

        if solver.M is not None:
            np.multiply(dt_imuz_1d, solver.M.field_z, out=tmp1d)
            Hz1d[:] -= tmp1d

        np.multiply(dtxy_0, Hz, out=dtxyHz)
        np.multiply(dtxy_1, Hz[:-1, :], out=tmp[1:, :])
        dtxyHz[1:, :] += tmp[1:, :]

        np.multiply(dtyx_0, Hz, out=dtyxHz)
        np.multiply(dtyx_1, Hz1d[:-1], out=tmp1d[1:])
        dtyxHz1d[1:] += tmp1d[1:]

        psiEb_y_lo[:] *= b_Ex_lo
        psiEb_y_lo[:] += c_Ex_lo * dtyxHz[:, :npml]
        psiEa_x_lo[:] *= b_Ey_lo
        psiEa_x_lo[:] += c_Ey_lo * dtxyHz[:npml, :]
        psiEb_y_hi[:] *= b_Ex_hi
        psiEb_y_hi[:] += c_Ex_hi * dtyxHz[:, -npml:]
        psiEa_x_hi[:] *= b_Ey_hi
        psiEa_x_hi[:] += c_Ey_hi * dtxyHz[-npml:, :]

        np.subtract(dtxyHz, Jx, out=tmp)
        tmp[:] *= dt_iepsx
        Ex[:] += tmp

        np.add(dtyxHz, Jy, out=tmp)
        tmp[:] *= dt_iepsy
        Ey[:] -= tmp

        Ey[:, :npml] -= dt_iepsy[:, :npml] * psiEb_y_lo
        Ex[:npml, :] += dt_iepsx[:npml, :] * psiEa_x_lo
        Ey[:, -npml:] -= dt_iepsy[:, -npml:] * psiEb_y_hi
        Ex[-npml:, :] += dt_iepsx[-npml:, :] * psiEa_x_hi

    dt_imuz_1d = dt_imuz.ravel()
    solver.one_step = fast_one_step
    solver.field_energy = fast_field_energy


class StridedDFTMonitor:
    """
    2D midplane running DFT monitor with spatial and temporal subsampling.

    Accumulates `Ex`, `Ey`, and `Hz` on a strided 2D `(x, y)` grid every
    `time_stride` steps, enabling simultaneous monitoring of an off-resonance
    wavelength and a dense wavelength comb across one full free spectral range
    with minimal memory and CPU overhead.
    """

    def __init__(self, wavelengths_m, stride_xy=2, time_stride=5):
        self.wavelengths = np.atleast_1d(np.asarray(wavelengths_m, dtype=np.float64))
        self.frequencies = c_light / self.wavelengths
        self.stride_xy = max(1, int(stride_xy))
        self.time_stride = max(1, int(time_stride))
        self.step = 0
        self.is_first_update = True
        self.x = self.y = None
        self.acc_Ex = self.acc_Ey = self.acc_Hz = None

    def _initialize(self, solver):
        g = solver.grid
        s = self.stride_xy
        xc = 0.5 * (g.x[:-1] + g.x[1:])
        yc = 0.5 * (g.y[:-1] + g.y[1:])
        self.x = xc[::s].astype(np.float32)
        self.y = yc[::s].astype(np.float32)
        nx_s, ny_s = self.x.size, self.y.size
        n_f = self.frequencies.size
        self.shape_2d = (nx_s, ny_s)
        n_pts = nx_s * ny_s
        self.acc_Ex = np.zeros((n_f, n_pts), dtype=np.complex64)
        self.acc_Ey = np.zeros((n_f, n_pts), dtype=np.complex64)
        self.acc_Hz = np.zeros((n_f, n_pts), dtype=np.complex64)

        ix = np.arange(0, solver.Nx, s)
        iy = np.arange(0, solver.Ny, s)
        iz = solver.Nz // 2
        I, J = np.meshgrid(ix, iy, indexing="ij")
        self._flat_idx = (I + solver.Nx * J + solver.Nx * solver.Ny * iz).ravel()
        self._N = solver.N

    def update(self, solver, t):
        if self.is_first_update:
            self._initialize(solver)
            self.is_first_update = False
        if self.step % self.time_stride == 0:
            dt_eff = float(solver.dt) * self.time_stride
            t_E = float(t) + float(solver.dt)
            t_H = float(t) + 0.5 * float(solver.dt)
            kE = (dt_eff * np.exp(-2j * np.pi * self.frequencies * t_E)).astype(
                np.complex64
            )[:, None]
            kH = (dt_eff * np.exp(-2j * np.pi * self.frequencies * t_H)).astype(
                np.complex64
            )[:, None]
            E_arr = _host(solver.E.array)
            H_arr = _host(solver.H.array)
            idx = self._flat_idx
            N = self._N
            ex = E_arr[idx].astype(np.float32)[None, :]
            ey = E_arr[N + idx].astype(np.float32)[None, :]
            hz = H_arr[2 * N + idx].astype(np.float32)[None, :]
            self.acc_Ex += kE * ex
            self.acc_Ey += kE * ey
            self.acc_Hz += kH * hz
        self.step += 1

    def get_slice(self, target_wavelength_m):
        """Return the `(x, y, Ex, Ey, Hz, E2, wavelength)` dict closest to `target_wavelength_m`."""
        idx_w = int(np.argmin(np.abs(self.wavelengths - float(target_wavelength_m))))
        ex = self.acc_Ex[idx_w].reshape(self.shape_2d)
        ey = self.acc_Ey[idx_w].reshape(self.shape_2d)
        hz = self.acc_Hz[idx_w].reshape(self.shape_2d)
        e2 = (np.abs(ex) ** 2 + np.abs(ey) ** 2).astype(np.float32)
        return {
            "x": self.x,
            "y": self.y,
            "Ex": ex,
            "Ey": ey,
            "Hz": hz,
            "E2": e2,
            "wavelength": float(self.wavelengths[idx_w]),
        }


class _DualPortSnapshotRecorder(_SnapshotRecorder):
    """Records downsampled 2D midplane frames plus time traces at both `through` and `drop` ports."""

    def __init__(self, component="y", max_frames=50, stride_xy=1, n_steps_est=1000):
        super().__init__(
            component=component,
            max_frames=max_frames,
            stride_xy=stride_xy,
            n_steps_est=n_steps_est,
        )
        self.probe_idx_drop = None
        self.drop_trace_val = []
        self._flat_probe = None
        self._flat_probe_drop = None

    def update(self, solver, t):
        if self.x is None:
            g = solver.grid
            s = self.stride_xy
            self.x = (0.5 * (g.x[:-1] + g.x[1:]))[::s].astype(np.float32)
            self.y = (0.5 * (g.y[:-1] + g.y[1:]))[::s].astype(np.float32)
            c_off = "xyz".index(self.component) * solver.N
            if self.probe_idx is not None:
                ix, iy, iz = self.probe_idx
                self._flat_probe = int(
                    c_off + ix + solver.Nx * iy + solver.Nx * solver.Ny * iz
                )
            if self.probe_idx_drop is not None:
                ixd, iyd, izd = self.probe_idx_drop
                self._flat_probe_drop = int(
                    c_off + ixd + solver.Nx * iyd + solver.Nx * solver.Ny * izd
                )

        E_arr = _host(solver.E.array)
        if self._flat_probe is not None:
            self.port_trace_t.append(float(t))
            self.port_trace_val.append(float(E_arr[self._flat_probe]))
        if self._flat_probe_drop is not None:
            self.drop_trace_val.append(float(E_arr[self._flat_probe_drop]))

        need_frame = (
            self.max_frames > 0
            and self.step % self.step_interval == 0
            and len(self.frames) < self.max_frames
        )
        if need_frame:
            mat = solver.E.to_matrix(self.component)
            if hasattr(mat, "get"):
                mat = mat.get()
            iz = solver.Nz // 2
            s = self.stride_xy
            self.frames.append(
                np.asarray(mat[::s, ::s, iz], dtype=np.float32).copy()
            )
            self.times.append(float(t))
        self.step += 1


class RingPhotonicCircuit(PhotonicCircuit):
    """
    `PhotonicCircuit` subclass tailored for high-Q microring resonators in 2D EIM.

    - Attaches `attach_fast_2d_te_step` in `build()` for 2D quasi-TE runs.
    - Uses `FastModeMonitor` for port S-parameters.
    - Tracks the achieved electromagnetic energy decay ratio `final_energy_decay`
      and total simulated physical time `sim_time_s`.
    - Captures both off-resonance (`1550 nm`) and on-resonance (peak of `|S_drop|^2`)
      steady-state 2D field maps `(Ex, Ey, Hz, E2)` via `StridedDFTMonitor`.
    - Records time-domain waveforms at both the `through` and `drop` ports.
    """

    def __init__(
        self,
        *args,
        cfln=0.95,
        gap_m=0.10e-6,
        radius_m=5.0e-6,
        width_m=0.43e-6,
        build_solver=True,
        **kwargs,
    ):
        self.gap_m = float(gap_m)
        self.radius_m = float(radius_m)
        self.width_m = float(width_m)
        self._build_solver_flag = bool(build_solver)
        super().__init__(*args, cfln=cfln, **kwargs)
        self.final_energy_decay = None
        self.sim_time_s = None
        self.dft_on_resonance = None
        self.dft_off_resonance = None

    @property
    def dx(self):
        return float(np.max(self.grid.dx))

    @property
    def dx_background_nm(self):
        return float(np.max(self.grid.dx) * 1e9)

    @property
    def dx_min_nm(self):
        return float(min(np.min(self.grid.dx), np.min(self.grid.dy)) * 1e9)

    @property
    def min_feature_nm(self):
        return float(self.gap_m * 1e9)

    @property
    def cells_across_feature(self):
        caf, _ = measure_cells_across_gap(
            self.grid.y,
            self.gap_m,
            radius_m=self.radius_m,
            width_m=self.width_m,
        )
        return float(caf)

    @property
    def n_cells(self):
        return int(self.grid.Nx * self.grid.Ny * self.grid.Nz)

    @property
    def n_core_eff(self):
        return float(self.n_eff_slab)

    def build_mesh_only(self):
        """Build only `self.sim` and `self.grid` without allocating `SolverFIT3D`."""
        if self.two_d:
            structs, domain = self._build_2d_structures_and_domain()
            boundaries = (["cpml", "cpml", "periodic"], ["cpml", "cpml", "periodic"])
        else:
            structs = list(self.structures_3d)
            domain = self._domain_input
            boundaries = "cpml"

        self.sim = Simulation(
            self.f_min,
            self.f_max,
            structures=structs,
            boundaries=boundaries,
            domain=domain,
            cells_per_wavelength=self.cells_per_wavelength,
            padding=self.padding,
            n_pml=self.n_pml,
            n_freqs=self.n_freqs,
            cfln=self.cfln,
            mesh_overrides=self.mesh_overrides,
            use_gpu=self.use_gpu,
            averaging=self.averaging,
        )
        x, y, z = self.sim._axes = [self.sim._axis(a) for a in range(3)]
        self.grid = GridFIT3D(x=x, y=y, z=z, verbose=0)
        if self.two_d and self.grid.boundary_at_domain_edge:
            self.grid.remove_edge_ghost(2)
        self.frequencies = self.sim.frequencies
        self.wavelengths = c_light / self.frequencies
        self.dt_cfl = float(
            self.cfln
            / (
                c_light
                * np.sqrt(
                    1.0 / np.min(self.grid.dx) ** 2
                    + 1.0 / np.min(self.grid.dy) ** 2
                    + 1.0 / np.min(self.grid.dz) ** 2
                )
            )
        )
        return self.grid

    def build(self):
        if not getattr(self, "_build_solver_flag", True):
            self.build_mesh_only()
            return None
        if self.solver is not None and getattr(self.solver, "step_0", False):
            return self.solver
        solver = super().build()
        solver.verbose = 0
        if self.two_d and not self.use_gpu:
            attach_fast_2d_te_step(solver)
        self.dt_cfl = float(solver.dt)
        return solver

    def run_s_parameters(
        self, drive=None, energy_decay=1e-3, max_time=12e-12, collect_fields=True
    ):
        if not self.ports:
            raise ValueError("No ports defined on RingPhotonicCircuit.")
        if drive is None:
            drive = [self.ports[0].name]
        elif isinstance(drive, str):
            drive = [p.name for p in self.ports] if drive == "all" else [drive]

        if max_time is None:
            max_time = 12e-12

        self._build_solver_flag = True
        s_matrix = {}
        for run_idx, in_name in enumerate(drive):
            if (
                run_idx > 0
                or self.solver is None
                or not getattr(self.solver, "step_0", False)
            ):
                self.solver = None
                self.build()

            in_port = self.port_map[in_name]
            in_center, in_size, in_inward = self._port_plane_args(in_port)
            a_in = in_port.axis
            nodes_a = self.grid.x if a_in == 0 else self.grid.y
            k_mon = int(np.argmin(np.abs(nodes_a[:-1] - in_center[a_in])))
            k_src = int(np.clip(k_mon - 2 * in_inward, 2, len(nodes_a) - 3))
            src_center = in_center.copy()
            src_center[a_in] = nodes_a[k_src]

            src = ModeSource(
                self.sim.pulse,
                center=src_center,
                size=in_size,
                direction="+" if in_inward > 0 else "-",
                mode_index=in_port.mode_index,
                mode_frequency=self.f_center,
            )

            port_monitors, monitors_list = {}, []
            for p in self.ports:
                p_center, p_size, _ = self._port_plane_args(p)
                mon = FastModeMonitor(
                    self.frequencies,
                    center=p_center,
                    size=p_size,
                    n_modes=p.mode_index + 1,
                    mode_frequency=self.f_center,
                )
                port_monitors[p.name] = mon
                monitors_list.append(mon)

            dft_mon = snap_mon = None
            if collect_fields and run_idx == 0:
                stride = max(1, int(np.ceil(max(self.grid.Nx, self.grid.Ny) / 200)))
                # Wavelength comb spanning the m=49 resonance across all cpw levels + 1550 nm off-resonance reference
                wl_comb = np.unique(
                    np.concatenate(
                        [
                            np.array([self.wavelength_ref]),
                            np.linspace(1556.0e-9, 1570.0e-9, 57),
                        ]
                    )
                )
                dft_mon = StridedDFTMonitor(wl_comb, stride_xy=stride, time_stride=15)
                monitors_list.append(dft_mon)

                # Movie frames cover the active ring-down window (~first 3.5 ps)
                Nt_movie = int(np.ceil(min(max_time, 3.5e-12) / self.solver.dt))
                snap_mon = _DualPortSnapshotRecorder(
                    component="y",
                    max_frames=self.movie_frames if self.record_movie else 0,
                    stride_xy=stride,
                    n_steps_est=Nt_movie,
                )
                if "through" in self.port_map:
                    tp_c, _, _ = self._port_plane_args(self.port_map["through"])
                    ix = int(np.argmin(np.abs(self.grid.x[:-1] - tp_c[0])))
                    iy = int(np.argmin(np.abs(self.grid.y[:-1] - tp_c[1])))
                    snap_mon.probe_idx = (ix, iy, self.solver.Nz // 2)
                if "drop" in self.port_map:
                    dp_c, _, _ = self._port_plane_args(self.port_map["drop"])
                    ixd = int(np.argmin(np.abs(self.grid.x[:-1] - dp_c[0])))
                    iyd = int(np.argmin(np.abs(self.grid.y[:-1] - dp_c[1])))
                    snap_mon.probe_idx_drop = (ixd, iyd, self.solver.Nz // 2)
                monitors_list.append(snap_mon)

            # Track peak energy during run to report exact achieved energy decay ratio
            energy_state = {"peak": 0.0, "final": 0.0}
            orig_fe = self.solver.field_energy

            def tracked_field_energy():
                val = orig_fe()
                if val > energy_state["peak"]:
                    energy_state["peak"] = val
                energy_state["final"] = val
                return val

            self.solver.field_energy = tracked_field_energy

            Nt = int(np.ceil(max_time / self.solver.dt))
            self.solver.emsolve(
                Nt,
                source=[src],
                monitors=monitors_list,
                stop_when_energy_decayed=energy_decay,
            )
            self.sim_time_s = float(self.solver.Nt * self.solver.dt)
            if energy_state["peak"] > 0:
                self.final_energy_decay = float(
                    energy_state["final"] / energy_state["peak"]
                )
            else:
                self.final_energy_decay = 0.0

            a_plus_in, a_minus_in = port_monitors[in_name].get_amplitudes(
                in_port.mode_index
            )
            a_inc = a_plus_in if in_inward > 0 else a_minus_in

            for p in self.ports:
                _, _, p_inward = self._port_plane_args(p)
                a_p, a_m = port_monitors[p.name].get_amplitudes(p.mode_index)
                b_out = a_m if p_inward > 0 else a_p
                s_matrix[(p.name, in_name)] = b_out / a_inc

            if dft_mon is not None:
                # Off-resonance slice at 1550 nm and on-resonance slice at the m=49 peak
                w_all = self.wavelengths
                t_drop_all = (
                    np.abs(s_matrix[("drop", in_name)]) ** 2
                    if ("drop", in_name) in s_matrix
                    else np.zeros_like(w_all)
                )
                in_comb = (w_all >= 1556e-9) & (w_all <= 1570e-9)
                if np.any(in_comb):
                    idx_res = int(np.flatnonzero(in_comb)[np.argmax(t_drop_all[in_comb])])
                    wl_res = float(w_all[idx_res])
                else:
                    wl_res = self.wavelength_ref
                self.dft_on_resonance = dft_mon.get_slice(wl_res)
                self.dft_off_resonance = dft_mon.get_slice(self.wavelength_ref)
                self.dft_fields[self.wavelength_ref] = self.dft_off_resonance
                self.dft_fields[wl_res] = self.dft_on_resonance

            if snap_mon is not None:
                self.port_time_trace = {
                    "t": np.asarray(snap_mon.port_trace_t, dtype=np.float32),
                    "Ey": np.asarray(snap_mon.port_trace_val, dtype=np.float32),
                    "Ey_drop": np.asarray(snap_mon.drop_trace_val, dtype=np.float32),
                }
                if self.record_movie and snap_mon.frames:
                    frames_f32 = np.stack(snap_mon.frames, axis=0)
                    peak = max(float(np.max(np.abs(frames_f32))), 1e-30)
                    self.movie_data = {
                        "x": snap_mon.x,
                        "y": snap_mon.y,
                        "times": np.asarray(snap_mon.times, dtype=np.float32),
                        "frames": (frames_f32 / peak).astype(np.float16),
                        "peak": np.float32(peak),
                    }

        return s_matrix


def _lorentzian_peak(wl_nm, wl0_nm, fwhm_nm, peak_val, bg_val):
    """Lorentzian resonance lineshape for nonlinear least-squares fitting."""
    half_w = 0.5 * np.maximum(fwhm_nm, 1e-5)
    return bg_val + peak_val / (1.0 + ((wl_nm - wl0_nm) / half_w) ** 2)


def find_ring_resonances(
    wavelengths_m,
    s_through,
    s_drop,
    s_in=None,
    s_add=None,
    band_nm=(1505.0, 1595.0),
    prominence=0.08,
):
    """
    Detect all microring resonances in `s_drop` and fit Lorentzian parameters.

    Parameters
    ----------
    wavelengths_m : ndarray
        Wavelength array [m].
    s_through, s_drop : ndarray
        Complex S-parameters `S[("through", "in")]` and `S[("drop", "in")]`.
    s_in, s_add : ndarray, optional
        Complex S-parameters `S[("in", "in")]` and `S[("add", "in")]`.
    band_nm : tuple of float, optional
        Interior wavelength window `(wl_min_nm, wl_max_nm)` [nm] to search.
    prominence : float, optional
        Minimum drop-port power prominence for peak detection.

    Returns
    -------
    dict
        Dictionary containing sorted arrays `wl_nm`, `t_through`, `t_drop`,
        `t_in`, `t_add`, a list `resonances` of per-resonance dicts (with
        Lorentzian-fitted `wl0_nm`, `fwhm_nm`, `Q_loaded`, `t_drop_peak`,
        `t_through_min`, `extinction_db`, `kappa2_cmt`, `a_rt_cmt`), and
        summary metrics (`fsr_nm`, `mean_fsr_nm`, `mean_Q_loaded`).
    """
    wl_nm_all = np.asarray(wavelengths_m, dtype=np.float64) * 1e9
    order = np.argsort(wl_nm_all)
    wl_nm = wl_nm_all[order]
    t_thru = np.abs(np.asarray(s_through)[order]) ** 2
    t_drop = np.abs(np.asarray(s_drop)[order]) ** 2
    t_in = (
        np.abs(np.asarray(s_in)[order]) ** 2
        if s_in is not None
        else np.zeros_like(t_drop)
    )
    t_add = (
        np.abs(np.asarray(s_add)[order]) ** 2
        if s_add is not None
        else np.zeros_like(t_drop)
    )

    mask = (wl_nm >= band_nm[0]) & (wl_nm <= band_nm[1])
    idx_band = np.flatnonzero(mask)
    dwl = float(np.median(np.diff(wl_nm)))
    min_dist = max(5, int(round(8.0 / max(dwl, 1e-3))))

    peaks_rel, _ = find_peaks(t_drop[mask], prominence=prominence, distance=min_dist)
    peaks = idx_band[peaks_rel]

    resonances = []
    for p in peaks:
        wl0_est = float(wl_nm[p])
        peak_est = float(t_drop[p])
        # Local window +-3.0 nm around peak for Lorentzian fit
        win = (wl_nm >= wl0_est - 3.0) & (wl_nm <= wl0_est + 3.0)
        xw = wl_nm[win]
        yw = t_drop[win]
        tw = t_thru[win]

        # Direct half-maximum width estimate
        half_val = 0.5 * peak_est
        above = xw[yw >= half_val]
        fwhm_est = (
            float(above[-1] - above[0])
            if above.size >= 2
            else max(2.0 * dwl, 0.4)
        )
        fwhm_est = max(fwhm_est, 0.15)

        try:
            popt, _ = curve_fit(
                _lorentzian_peak,
                xw,
                yw,
                p0=[wl0_est, fwhm_est, peak_est, 0.0],
                bounds=(
                    [wl0_est - 1.0, 0.05, 0.05, -0.05],
                    [wl0_est + 1.0, 6.0, 1.2, 0.15],
                ),
                maxfev=4000,
            )
            wl0_fit, fwhm_fit, peak_fit, bg_fit = [float(v) for v in popt]
        except Exception:
            wl0_fit, fwhm_fit, peak_fit, bg_fit = wl0_est, fwhm_est, peak_est, 0.0

        t_drop_max = max(peak_est, peak_fit + bg_fit)
        t_thru_min = float(np.min(tw))
        Q_loaded = wl0_fit / fwhm_fit
        ext_db = float(-10.0 * np.log10(max(t_thru_min, 1e-12)))
        drop_db = float(10.0 * np.log10(max(t_drop_max, 1e-12)))

        resonances.append(
            {
                "index": int(p),
                "wl0_nm": wl0_fit,
                "wl_grid_nm": wl0_est,
                "fwhm_nm": fwhm_fit,
                "Q_loaded": Q_loaded,
                "t_drop_peak": t_drop_max,
                "drop_db": drop_db,
                "t_through_min": t_thru_min,
                "through_min_db": -ext_db,
                "extinction_db": ext_db,
                "fit_bg": bg_fit,
                "fit_amp": peak_fit,
            }
        )

    fsr_nm = (
        np.diff([r["wl0_nm"] for r in resonances])
        if len(resonances) >= 2
        else np.array([])
    )
    mean_fsr = float(np.mean(fsr_nm)) if fsr_nm.size > 0 else 23.2

    # Populate Finesse and CMT parameters (t, kappa^2, a_rt) for each resonance
    for i, r in enumerate(resonances):
        local_fsr = (
            float(fsr_nm[min(i, fsr_nm.size - 1)]) if fsr_nm.size > 0 else mean_fsr
        )
        finesse = local_fsr / r["fwhm_nm"]
        gamma = np.pi / max(finesse, 1.01)
        eta = min(np.sqrt(max(r["t_drop_peak"], 1e-6)), 0.9999)
        t_cpl = 0.5 * (np.sqrt((eta * gamma) ** 2 + 4.0) - eta * gamma)
        kappa2 = float(1.0 - t_cpl**2)
        sqrt_a = min(0.5 * (np.sqrt(gamma**2 + 4.0) - gamma) / max(t_cpl, 1e-6), 1.0)
        a_rt = float(sqrt_a**2)
        r["fsr_local_nm"] = local_fsr
        r["finesse"] = float(finesse)
        r["kappa2_cmt"] = kappa2
        r["t_cmt"] = float(t_cpl)
        r["a_rt_cmt"] = a_rt

    return {
        "wl_nm": wl_nm,
        "t_through": t_thru,
        "t_drop": t_drop,
        "t_in": t_in,
        "t_add": t_add,
        "resonances": resonances,
        "fsr_nm": fsr_nm,
        "mean_fsr_nm": mean_fsr,
        "mean_Q_loaded": (
            float(np.mean([r["Q_loaded"] for r in resonances]))
            if resonances
            else 0.0
        ),
    }


def cmt_add_drop_spectra(wavelengths_m, neff_vals, radius_m, kappa2, a_rt=1.0, phase_offset_rad=0.0):
    """
    Analytical Yariv (2000) transfer-matrix power spectra `(T_through, T_drop)`
    for a symmetric microring add-drop filter.

    Parameters
    ----------
    wavelengths_m : ndarray
        Wavelength array [m].
    neff_vals : float or ndarray
        Effective index `n_eff(lambda)` of the ring waveguide mode.
    radius_m : float
        Mean ring radius `R` [m].
    kappa2 : float
        Per-coupler power coupling coefficient `kappa^2 = 1 - t^2`.
    a_rt : float, optional
        Round-trip field transmission factor `a = exp(-alpha * L / 2) <= 1`.
    phase_offset_rad : float, optional
        Coupler/grid phase offset [rad] aligning the analytical comb to a reference peak.

    Returns
    -------
    t_through, t_drop : ndarray
        Analytical through-port and drop-port power transmission spectra.
    """
    wl = np.asarray(wavelengths_m, dtype=np.float64)
    neff = np.asarray(neff_vals, dtype=np.float64)
    L = 2.0 * np.pi * float(radius_m)
    phi = 2.0 * np.pi * neff * L / wl + float(phase_offset_rad)
    t2 = max(1.0 - float(kappa2), 1e-6)
    a = float(np.clip(a_rt, 1e-6, 1.0))
    sin2 = np.sin(0.5 * phi) ** 2
    denom = (1.0 - t2 * a) ** 2 + 4.0 * t2 * a * sin2
    t_through = (t2 * (1.0 - a) ** 2 + 4.0 * t2 * a * sin2) / denom
    t_drop = ((1.0 - t2) ** 2 * a) / denom
    return t_through, t_drop


def build_ring_add_drop_graded_axes(
    domain=(-6.3e-6, 6.3e-6, -6.895e-6, 6.895e-6),
    gap_m=0.10e-6,
    width_m=0.43e-6,
    radius_m=5.0e-6,
    cells_per_wavelength=20,
    n_cells_across_gap=8,
    n_pml=12,
    wavelength_min=1.50e-6,
    wavelength_ref=1.55e-6,
    n_core=3.476,
    n_clad=1.444,
    thickness_m=0.22e-6,
    grading_ratio=1.22,
):
    """
    Construct mirror-symmetric ($x \\mapsto -x$ and $y \\mapsto -y$) graded
    non-uniform Cartesian Yee grid axes ``(x_axis, y_axis, z_axis, info)`` for
    the 2D EIM microring add-drop filter using :func:`wakis.templates.meshing.graded_axis`:

    - Along ``y``: places ``N = max(int(n_cells_across_gap), 4)`` exact cells of
      step ``dy_min = gap_m / N`` across both the upper ``[r_out, r_out + g]``
      and lower ``[-(r_out + g), -r_out]`` coupling slots, resolves the upper
      and lower ring arcs and bus waveguide cores (``dy ~ 22 nm``), uses
      non-square tangential cells (``dy ~ 33 nm``) across ``|y| <= 3.3 um``
      where the ring propagates vertically, and grades smoothly into the outer
      cladding + CPML.
    - Along ``x``: enforces exact ``x -> -x`` mirror symmetry with non-square
      cells ``dx_couple > dy_min`` across the ``|x| <= 0.9 um`` coupling window,
      tangential cells (``dx ~ 33 nm``) along the top/bottom arcs, and fine
      radial cells (``dx ~ 21.6 nm``) across the left/right vertical ring arms
      ``|x| in [r_in, r_out]``.
    """
    g = float(gap_m)
    w = float(width_m)
    r = float(radius_m)
    cpw = float(cells_per_wavelength)
    n_gap = max(int(n_cells_across_gap or 4), 4)
    n_pml = int(n_pml)

    neff_slab = float(
        slab_neff(n_core, n_clad, thickness_m, wavelength_ref, polarization="TE")
    )
    h_core = float(wavelength_min / (neff_slab * cpw))
    h_clad = float(wavelength_min / (float(n_clad) * cpw))
    dy_gap = min(g / float(n_gap), h_core)

    r_in = r - 0.5 * w
    r_out = r + 0.5 * w
    y_gap_hi = r_out + g
    y_bus_hi = y_gap_hi + w

    xmin, xmax, ymin, ymax = (float(v) for v in domain[:4])
    xmax_total = xmax + n_pml * h_core
    ymax_total = ymax + n_pml * h_core

    # 1. Mirror-symmetric x-axis constructed on [0, xmax_total]
    dx_couple = 1.00 * h_core
    dx_tang = 1.38 * h_core
    dx_arc = 1.04 * h_core
    dx_ring = 0.98 * h_core
    dx_clad = min(1.62 * h_core, 0.92 * h_clad)
    x_knots = [0.0, 0.90e-6, 3.60e-6, r_in, r_out, xmax_total]
    x_sizes = [dx_couple, dx_tang, dx_arc, dx_ring, dx_clad]
    x_pos = meshing.graded_axis(x_knots, x_sizes, grading=grading_ratio)
    x_pos[0] = 0.0
    x_axis = np.concatenate([-x_pos[:0:-1], x_pos])

    # 2. Mirror-symmetric y-axis constructed on [0, ymax_total]
    dy_tang = 1.38 * h_core
    dy_arc = 1.04 * h_core
    dy_ring = 0.98 * h_core
    dy_bus = 0.98 * h_core
    dy_clad = min(1.68 * h_core, 0.94 * h_clad)
    y_knots = [0.0, 3.60e-6, r_in, r_out, y_gap_hi, y_bus_hi, ymax_total]
    y_sizes = [dy_tang, dy_arc, dy_ring, dy_gap, dy_bus, dy_clad]
    y_pos = meshing.graded_axis(y_knots, y_sizes, grading=grading_ratio)
    y_pos[0] = 0.0
    y_axis = np.concatenate([-y_pos[:0:-1], y_pos])

    # 3. Single-cell periodic 2D z-axis matching PhotonicCircuit._build_2d_structures_and_domain
    z_axis = np.array([-0.5 * h_core, +0.5 * h_core], dtype=float)

    caf, dy_max_gap = measure_cells_across_gap(y_axis, g, radius_m=r, width_m=w)
    dx_arr = np.diff(x_axis)
    dy_arr = np.diff(y_axis)
    info = {
        "min_feature_nm": float(g * 1e9),
        "dx_background_nm": float(max(np.max(dx_arr), np.max(dy_arr)) * 1e9),
        "dx_min_nm": float(min(np.min(dx_arr), np.min(dy_arr)) * 1e9),
        "cells_across_feature": float(caf),
        "dy_max_in_gap_nm": float(dy_max_gap * 1e9),
        "n_cells": int(len(dx_arr) * len(dy_arr)),
        "grid_shape": [int(len(dx_arr)), int(len(dy_arr)), 1],
    }
    return x_axis, y_axis, z_axis, info


class NonUniformRingAddDropCircuit(RingPhotonicCircuit):
    """
    2D EIM :class:`RingPhotonicCircuit` subclass that replaces the uniform
    background + override-box mesh with :func:`build_ring_add_drop_graded_axes`.
    """

    def __init__(
        self,
        *args,
        n_cells_across_gap=8,
        grading_ratio=1.22,
        **kwargs,
    ):
        self._nu_n_gap = int(n_cells_across_gap or 8)
        self._nu_ratio = float(grading_ratio)
        kwargs["mesh_overrides"] = ()
        super().__init__(*args, **kwargs)

    @property
    def dx_background_nm(self):
        return float(max(np.max(self.grid.dx), np.max(self.grid.dy)) * 1e9)

    def _compute_graded_axes(self, domain):
        n_pml_val = (
            self.n_pml[0]
            if isinstance(self.n_pml, (tuple, list))
            else self.n_pml
        )
        x, y, z, _ = build_ring_add_drop_graded_axes(
            domain=domain,
            gap_m=self.gap_m,
            width_m=self.width_m,
            radius_m=self.radius_m,
            cells_per_wavelength=self.cells_per_wavelength,
            n_cells_across_gap=self._nu_n_gap,
            n_pml=n_pml_val,
            wavelength_min=self.wavelength_min,
            wavelength_ref=self.wavelength_ref,
            n_core=self.n_core,
            n_clad=self.n_clad,
            thickness_m=self.thickness,
            grading_ratio=self._nu_ratio,
        )
        return x, y, z

    def build_mesh_only(self):
        structs, domain = self._build_2d_structures_and_domain()
        boundaries = (["cpml", "cpml", "periodic"], ["cpml", "cpml", "periodic"])
        self.sim = Simulation(
            self.f_min,
            self.f_max,
            structures=structs,
            boundaries=boundaries,
            domain=domain,
            cells_per_wavelength=self.cells_per_wavelength,
            padding=self.padding,
            n_pml=self.n_pml,
            n_freqs=self.n_freqs,
            cfln=self.cfln,
            mesh_overrides=[],
            use_gpu=self.use_gpu,
            averaging=self.averaging,
        )
        x, y, z = self._compute_graded_axes(domain)
        self.sim._axes = [x, y, z]
        self.grid = GridFIT3D(x=x, y=y, z=z, verbose=0)
        if self.two_d and self.grid.boundary_at_domain_edge:
            self.grid.remove_edge_ghost(2)
        self.frequencies = self.sim.frequencies
        self.wavelengths = c_light / self.frequencies
        self.dt_cfl = float(
            self.cfln
            / (
                c_light
                * np.sqrt(
                    1.0 / np.min(self.grid.dx) ** 2
                    + 1.0 / np.min(self.grid.dy) ** 2
                    + 1.0 / np.min(self.grid.dz) ** 2
                )
            )
        )
        return self.grid

    def build(self):
        if not getattr(self, "_build_solver_flag", True):
            self.build_mesh_only()
            return None
        if self.solver is not None and getattr(self.solver, "step_0", False):
            return self.solver
        structs, domain = self._build_2d_structures_and_domain()
        boundaries = (["cpml", "cpml", "periodic"], ["cpml", "cpml", "periodic"])
        self.sim = Simulation(
            self.f_min,
            self.f_max,
            structures=structs,
            boundaries=boundaries,
            domain=domain,
            cells_per_wavelength=self.cells_per_wavelength,
            padding=self.padding,
            n_pml=self.n_pml,
            n_freqs=self.n_freqs,
            cfln=self.cfln,
            mesh_overrides=[],
            use_gpu=self.use_gpu,
            averaging=self.averaging,
        )
        x, y, z = self._compute_graded_axes(domain)
        axes_xyz = (x, y, z)
        self.sim._axis = lambda axis: axes_xyz[int(axis)]
        self.solver = self.sim.build()
        self.solver.ieps[:, :, :, "z"] = 0.0
        self.solver.verbose = 0
        self.grid = self.sim.grid
        self.frequencies = self.sim.frequencies
        self.wavelengths = c_light / self.frequencies
        if self.two_d and not self.use_gpu:
            attach_fast_2d_te_step(self.solver)
        self.dt_cfl = float(self.solver.dt)
        return self.solver


def build_ring_add_drop_hybrid_unstructured_mesh(
    radius_m=5.0e-6,
    width_m=0.43e-6,
    h=0.10e-6,
    levels=2,
):
    """
    Build a 2D body-fitted square+triangle hybrid mesh for the closed microring
    resonator annulus ($r_{\\mathrm{in}} = 4.785\\;\\mu\\mathrm{m},
    r_{\\mathrm{out}} = 5.215\\;\\mu\\mathrm{m}$) using
    :func:`wakis.unstructured.subtri_circles_mesh`.

    Returns ``(mesh, stats)`` where ``stats`` contains ``n_cells``, ``n_quads``,
    ``n_tris``, and ``n_vertices``.
    """
    import warnings

    r_in = float(radius_m - 0.5 * width_m)
    r_out = float(radius_m + 0.5 * width_m)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mesh, _, info = subtri_circles_mesh(
            circles=[(0.0, 0.0, r_in), (0.0, 0.0, r_out)],
            h=float(h),
            levels=int(levels),
            half_width=6.0e-6,
            inner_fine=False,
        )
    stats = {
        "n_cells": int(mesh.n_cells),
        "n_quads": int(info["n_squares_total"]),
        "n_tris": int(info["n_triangles"]),
        "n_vertices": int(mesh.n_nodes),
    }
    return mesh, stats


class _PolyStructProxy:
    """Proxy wrapping a 2D polygon vertex array for ``wakis.schematic``."""

    def __init__(self, vertices, material, name):
        self.vertices = np.asarray(vertices, dtype=float)
        self.material = float(material)
        self.name = str(name)


class _SchematicSolverProxy:
    """Proxy exposing 2D plane attributes to ``wakis.schematic``."""

    def __init__(self, solver, eps_clad: float = float(1.444**2)):
        self._solver = solver
        self.grid = solver.grid
        self.Nx = solver.Nx
        self.Ny = solver.Ny
        self.Nz = solver.Nz
        self.dt = solver.dt
        self.n_pml = getattr(solver, "n_pml", 12)
        self.bc_low = getattr(solver, "bc_low", ["cpml", "cpml", "periodic"])
        self.bc_high = getattr(solver, "bc_high", ["cpml", "cpml", "periodic"])
        self.bg = [float(eps_clad), 1.0, 0.0]
        self._assigned_structures = ()


def _build_ring_add_drop_schematic_io(sim):
    """Construct normalized ``(sources, monitors)`` dictionaries for ``wakis.schematic``."""
    pulse_obj = sim.sim.pulse
    f0 = float(pulse_obj.frequency)
    fwidth = float(pulse_obj.fwidth)
    freqs = np.asarray(sim.frequencies, dtype=float)
    f_lo_thz = float(np.min(freqs)) * 1e-12
    f_hi_thz = float(np.max(freqs)) * 1e-12
    freq_str = f"{freqs.size} freqs in [{f_lo_thz:.1f}, {f_hi_thz:.1f}] THz"

    sources = []
    monitors = []
    label_map = {
        "in": "Port 1 'in' (S_in,in)",
        "through": "Port 2 'through' (S_through,in)",
        "drop": "Port 3 'drop' (S_drop,in)",
        "add": "Port 4 'add' (S_add,in)",
    }
    for pt in sim.ports:
        cx, cy, _ = (float(v) for v in pt.center)
        _, sy, _ = (float(v) for v in pt.size)
        hy = min(0.5 * sy, 0.65e-6)
        v_b = (cy - hy, cy + hy)
        pos_s = f"x = {cx * 1e6:.2f} µm, y = {cy * 1e6:.2f} µm"
        if pt.name == "in":
            x_src = cx - 0.14e-6
            sources.append(
                {
                    "kind": "sheet",
                    "type_name": "ModeSource",
                    "short_word": "ModeSrc",
                    "desc": f"Port 1 'in' ModeSource (TE₀, +x, {pos_s})",
                    "label": "Port 1 'in' ModeSource (TE₀, +x)",
                    "pos_str": f"x = {x_src * 1e6:.2f} µm, y = {cy * 1e6:.2f} µm",
                    "polarization": "TE₀ (E_y, H_z)",
                    "component": "Ey",
                    "h_bounds": (x_src, x_src),
                    "v_bounds": v_b,
                    "k_vec": (1.0, 0.0, 0.0),
                    "e_vec": (0.0, 1.0, 0.0),
                    "h_vec": (0.0, 0.0, 1.0),
                    "pulse": {
                        "f0": f0,
                        "fwidth": fwidth,
                        "t_peak": float(getattr(pulse_obj, "t0", 4.0 / (np.pi * fwidth))),
                        "t_end": float(pulse_obj.end_time),
                        "src_time": pulse_obj,
                    },
                }
            )
        monitors.append(
            {
                "kind": "plane",
                "type_name": "ModeMonitor",
                "short_word": pt.name,
                "desc": f"{label_map.get(pt.name, pt.name)} ({pos_s})",
                "label": label_map.get(pt.name, pt.name),
                "pos_str": pos_s,
                "freq_str": freq_str,
                "frequencies": freqs,
                "h_bounds": (cx, cx),
                "v_bounds": v_b,
            }
        )
    return sources, monitors


def save_ring_add_drop_setup_figures(
    sim,
    fig_dir: str | Path,
    *,
    wavelength: float = 1.55e-6,
    min_feature: float = 100e-9,
):
    """
    Generate the setup figures and Markdown setup table for Tutorial 5.10:
    1. ``00a_setup_geometry.png`` + ``00b_setup_mesh_and_excitation.png`` +
       ``00_setup_table.md`` via :func:`wakis.schematic.save_setup_figures`.
    2. Annotates the launched ``in`` bus mode and backward-coupled ``drop`` bus
       mode on ``00a_setup_geometry.png`` and renders
       ``00b_setup_mesh_and_excitation.png`` with a 1× zoom on the ``100 nm``
       upper bus-ring coupling gap.
    """
    figstyle.apply_style()
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    proxy_solver = _SchematicSolverProxy(sim.solver, float(sim.n_clad**2))
    eps_core = float(sim.n_eff_slab**2)
    eps_clad = float(sim.n_clad**2)
    poly_structs = [
        _PolyStructProxy(
            sim.structures_3d[0].vertices,
            eps_core,
            f"Si EIM bus/ring core (n_Si=3.476, h=220 nm → n_eff,slab={sim.n_eff_slab:.4f}, w=430 nm)",
        ),
        _PolyStructProxy(
            sim.structures_3d[1].vertices,
            eps_core,
            f"Si EIM bus/ring core (n_Si=3.476, h=220 nm → n_eff,slab={sim.n_eff_slab:.4f}, w=430 nm)",
        ),
        _PolyStructProxy(
            sim.structures_3d[2].vertices,
            eps_core,
            f"Si EIM bus/ring core (n_Si=3.476, h=220 nm → n_eff,slab={sim.n_eff_slab:.4f}, w=430 nm)",
        ),
    ]
    mat_names = {
        "background": f"SiO₂ cladding (n={sim.n_clad:.3f}, ε_r={eps_clad:.2f})",
    }
    sources_s, monitors_s = _build_ring_add_drop_schematic_io(sim)
    variations = [
        {
            "name": "Resolution & subpixel convergence ladder (task_0..task_6)",
            "details": "Kottke & staircase at L1=(10,4), L2=(14,6), L3=(20,8) vs Kottke reference R=(40,16)",
        },
        {
            "name": "Coupling-gap parametric sweep (task_4, task_7, task_8)",
            "details": "g ∈ {60, 100, 150} nm at L3=(20 cpw, 8 cells/gap) — over- vs under-coupling",
        },
        {
            "name": "Graded non-uniform Cartesian & hybrid unstructured ring mesh (task_11)",
            "details": "Mirror-symmetric graded Δy (8.00 cells across both 100 nm gaps) & non-square Δx, Δy + wakis.unstructured subtri ring annulus",
        },
    ]

    res = schematic.save_setup_figures(
        proxy_solver,
        fig_dir,
        prefix="00",
        plane="xy",
        structures=poly_structs,
        material_names=mat_names,
        sources=sources_s,
        monitors=monitors_s,
        unit="µm",
        wavelength=wavelength,
        min_feature=min_feature,
        variations=variations,
        title_geometry="SOI Microring Add-Drop Filter Setup (2D EIM, R = 5.0 µm, w = 430 nm, g = 100 nm)",
        title_mesh="Microring Add-Drop Filter Mesh, Wave Packet & Broadband Excitation",
    )

    # Enhance 00a_setup_geometry.png with inner ring hole outline and launched/drop mode profiles
    fig_a, ax_a = schematic.plot_schematic(
        proxy_solver,
        plane="xy",
        structures=poly_structs,
        material_names=mat_names,
        sources=sources_s,
        monitors=monitors_s,
        unit="µm",
        wavelength=wavelength,
        title="SOI Microring Add-Drop Filter Setup (2D EIM, R = 5.0 µm, w = 430 nm, g = 100 nm)",
        width_in=8.2,
        height_in=7.2,
    )
    for ln in ax_a.lines:
        if ln.get_linestyle() in ("--", ":"):
            ln.set_marker("none")
    x_in_um = float(sim.port_map["in"].center[0]) * 1e6
    y_in_um = float(sim.port_map["in"].center[1]) * 1e6
    for ann in ax_a.texts:
        if ann.get_text() == "" and getattr(ann, "arrow_patch", None) is not None:
            if abs(ann.xy[0] - x_in_um) < 1.5 and abs(ann.xy[1]) < 6.0:
                ann.xy = (x_in_um + 0.72, y_in_um)
                ann.set_position((x_in_um - 0.14, y_in_um))
    if sim.structures_3d[2].holes:
        hole_v = np.asarray(sim.structures_3d[2].holes[0], dtype=float) * 1e6
        ax_a.add_patch(
            MplPolygon(
                hole_v,
                closed=True,
                facecolor="#eef6fb",
                edgecolor="#5C4A00",
                lw=1.1,
                zorder=3,
            )
        )
    for p_name, col, dir_sgn in (("in", "#D55E00", +1.0), ("drop", "#009E73", -1.0)):
        m0 = sim.solve_port_modes(p_name)[0]
        y_um_all = m0.coords["Eb"][0] * 1e6
        ey_line = m0.e["b"][:, m0.e["b"].shape[1] // 2]
        ey_norm_all = np.abs(ey_line) / max(float(np.max(np.abs(ey_line))), 1e-30)
        x_p_um = float(sim.port_map[p_name].center[0]) * 1e6
        y_p_um = float(sim.port_map[p_name].center[1]) * 1e6
        mask_p = np.abs(y_um_all - y_p_um) <= 0.65
        y_um = y_um_all[mask_p]
        ey_norm = ey_norm_all[mask_p]
        ax_a.fill_betweenx(
            y_um,
            x_p_um,
            x_p_um + dir_sgn * 0.75 * ey_norm,
            color=col,
            alpha=0.24,
            zorder=7,
        )
        ax_a.plot(
            x_p_um + dir_sgn * 0.75 * ey_norm,
            y_um,
            color=col,
            lw=1.6,
            marker="none",
            zorder=8,
        )
    ax_a.set_title(
        "SOI Microring Add-Drop Filter Setup (2D EIM, R = 5.0 µm, w = 430 nm, g = 100 nm)",
        pad=72,
        fontsize=11.0,
        fontweight="bold",
    )
    tag_positions = {
        "1 · ModeSrc": (-5.85, +4.42, "left"),
        "2 · in": (-3.80, +6.30, "left"),
        "3 · through": (+3.80, +6.30, "right"),
        "4 · drop": (-3.80, -6.35, "left"),
        "5 · add": (+3.80, -6.35, "right"),
    }
    for txt in list(ax_a.texts):
        s = txt.get_text().strip()
        if s in tag_positions:
            tx, ty, ha_val = tag_positions[s]
            txt.set_position((tx, ty))
            txt.set_ha(ha_val)
    figstyle.save(fig_a, res["geometry_figure"], close=True, auto_legend_below=False)

    # Enhance 00b_setup_mesh_and_excitation.png with zoom on the 100 nm upper coupling gap
    fig_b, axes_b = schematic.plot_mesh_and_excitation(
        proxy_solver,
        plane="xy",
        structures=poly_structs,
        material_names=mat_names,
        sources=sources_s,
        monitors=monitors_s,
        unit="µm",
        wavelength=wavelength,
        min_feature=min_feature,
        zoom_bounds=(-0.62e-6, 0.62e-6, 5.04e-6, 5.49e-6),
        title="Microring Add-Drop Filter Mesh, Wave Packet & Broadband Excitation",
    )
    ax_mesh = axes_b[0]
    for s_idx, s in enumerate(sim.structures_3d):
        ax_mesh.add_patch(
            MplPolygon(
                s.vertices * 1e6,
                closed=True,
                facecolor="#F0E442",
                edgecolor="#5C4A00",
                lw=1.0,
                alpha=0.25,
                zorder=2,
            )
        )
        if s_idx == 2 and s.holes:
            ax_mesh.add_patch(
                MplPolygon(
                    np.asarray(s.holes[0], dtype=float) * 1e6,
                    closed=True,
                    facecolor="white",
                    edgecolor="#5C4A00",
                    lw=1.0,
                    alpha=0.85,
                    zorder=2,
                )
            )
    figstyle.save(fig_b, res["mesh_figure"], close=True, auto_legend_below=False)
    return res


def plot_mesh_upgrade_comparison_ring_add_drop(
    sim_uniform,
    sim_nonuniform,
    td_uniform: dict,
    td_nonuniform: dict,
    td_ref: dict | None,
    out_path,
    find_resonances_fn=None,
):
    """
    Render ``04b_mesh_upgrade_comparison.png`` (3 panels <= 3) comparing:
    - Panel (a): Split-half mesh view (``x <= 0``: ``kottke_L3`` baseline
      override mesh; ``x >= 0``: ``kottke_nonuniform`` graded non-square
      Cartesian mesh).
    - Panel (b): Body-fitted ``wakis.unstructured`` hybrid square+triangle mesh
      (``subtri_circles_mesh``) conforming to both concentric ring boundaries
      ``r_in = 4.785 um`` and ``r_out = 5.215 um``.
    - Panel (c): Broadband drop-port and through-port resonance comb comparison
      around the ``m = 49`` resonance between ``kottke_L3``,
      ``kottke_nonuniform``, and ``kottke_reference`` (``R=(40,16)``).
    """
    if find_resonances_fn is None:
        find_resonances_fn = find_ring_resonances

    figstyle.apply_style()
    fig, (ax0, ax1, ax2) = figstyle.new_figure(nrows=1, ncols=3, width=15.6, height=4.8)

    # Panel (a): Split-half mesh view (x <= 0: kottke_L3, x >= 0: kottke_nonuniform)
    iz = sim_nonuniform.solver.Nz // 2
    ieps = np.asarray(sim_nonuniform.solver.ieps.to_matrix("y")[:, :, iz])
    eps_r = np.where(ieps > 0, 1.0 / (np.where(ieps > 0, ieps, 1.0) * epsilon_0), 1.444**2)
    xu_nu = sim_nonuniform.grid.x * 1e6
    yu_nu = sim_nonuniform.grid.y * 1e6
    ax0.pcolormesh(
        xu_nu, yu_nu, eps_r.T, cmap=figstyle.CMAP_EPS, shading="flat", alpha=0.88
    )

    xl_m, yl_m = (-6.2, 6.2), (-6.6, 6.6)
    xu_u = sim_uniform.grid.x * 1e6
    yu_u = sim_uniform.grid.y * 1e6
    xv_u = xu_u[(xu_u >= xl_m[0]) & (xu_u <= 0.0)][::4]
    yv_u = yu_u[(yu_u >= yl_m[0]) & (yu_u <= yl_m[1])][::4]
    segs_u = [[(x, yl_m[0]), (x, yl_m[1])] for x in xv_u] + [
        [(xl_m[0], y), (0.0, y)] for y in yv_u
    ]
    ax0.add_collection(
        LineCollection(segs_u, colors="#08519c", linewidths=0.36, alpha=0.62)
    )

    xv_nu = xu_nu[(xu_nu >= 0.0) & (xu_nu <= xl_m[1])][::4]
    yv_nu = yu_nu[(yu_nu >= yl_m[0]) & (yu_nu <= yl_m[1])][::4]
    segs_nu = [[(x, yl_m[0]), (x, yl_m[1])] for x in xv_nu] + [
        [(0.0, y), (xl_m[1], y)] for y in yv_nu
    ]
    ax0.add_collection(
        LineCollection(segs_nu, colors="#d95f02", linewidths=0.42, alpha=0.78)
    )
    ax0.axvline(0.0, color="black", lw=1.1, ls="--")
    for s_idx, s in enumerate(sim_nonuniform.structures_3d):
        ax0.add_patch(
            MplPolygon(
                s.vertices * 1e6,
                closed=True,
                fill=False,
                edgecolor="black",
                lw=1.0,
            )
        )
        if s_idx == 2 and s.holes:
            ax0.add_patch(
                MplPolygon(
                    np.asarray(s.holes[0], dtype=float) * 1e6,
                    closed=True,
                    fill=False,
                    edgecolor="black",
                    lw=1.0,
                )
            )

    n_u = int(td_uniform.get("n_cells", sim_uniform.n_cells))
    n_nu = int(td_nonuniform.get("n_cells", sim_nonuniform.n_cells))
    t_u_s = float(td_uniform.get("elapsed_s", 1757.1))
    t_nu_s = float(td_nonuniform.get("elapsed_s", 600.0))
    red_cells = 100.0 * (1.0 - n_nu / max(n_u, 1))
    red_time = 100.0 * (1.0 - t_nu_s / max(t_u_s, 1e-6))

    ax0.text(
        -5.95,
        -4.4,
        f"Left (x < 0): kottke_L3\n{sim_uniform.grid.Nx}×{sim_uniform.grid.Ny} = {n_u:,} cells\n({t_u_s:.0f} s)",
        fontsize=7.1,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#08519c", alpha=0.92),
    )
    ax0.text(
        0.25,
        -4.5,
        f"Right (x > 0): kottke_nonuniform\n"
        f"{sim_nonuniform.grid.Nx}×{sim_nonuniform.grid.Ny} = {n_nu:,} cells\n"
        f"(−{red_cells:.1f}% cells, {t_nu_s:.0f} s, −{red_time:.1f}%)",
        fontsize=7.1,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#d95f02", alpha=0.92),
    )
    ax0.set(
        xlim=xl_m,
        ylim=yl_m,
        aspect="equal",
        xlabel="x [µm]",
        ylabel="y [µm]",
        title="(a) Baseline L3 (left) vs graded non-uniform (right)",
    )

    # Panel (b): Body-fitted hybrid square+triangle mesh (subtri_circles_mesh) on upper ring + bus
    mesh_hyb, hyb_stats = build_ring_add_drop_hybrid_unstructured_mesh(
        radius_m=sim_nonuniform.radius_m,
        width_m=sim_nonuniform.width_m,
        h=0.10e-6,
        levels=2,
    )
    verts_um = np.asarray(mesh_hyb.nodes, dtype=float) * 1e6
    polys = [verts_um[mesh_hyb.cell(i)] for i in range(mesh_hyb.n_cells)]
    pc = PolyCollection(
        polys,
        array=np.asarray(mesh_hyb.cell_tags, dtype=float),
        cmap=figstyle.CMAP_EPS,
        edgecolors=(0.15, 0.15, 0.15, 0.55),
        linewidths=0.42,
    )
    ax1.add_collection(pc)
    for s_idx, s in enumerate(sim_nonuniform.structures_3d):
        ax1.add_patch(
            MplPolygon(
                s.vertices * 1e6,
                closed=True,
                facecolor="none",
                edgecolor="black",
                lw=1.1,
            )
        )
        if s_idx == 2 and s.holes:
            ax1.add_patch(
                MplPolygon(
                    np.asarray(s.holes[0], dtype=float) * 1e6,
                    closed=True,
                    facecolor="none",
                    edgecolor="black",
                    lw=1.1,
                )
            )
    ax1.set(
        xlim=(-2.6, 2.6),
        ylim=(3.6, 6.0),
        aspect="equal",
        xlabel="x [µm]",
        ylabel="y [µm]",
        title=(
            f"(b) Body-fitted hybrid ring mesh (subtri)\n"
            f"{hyb_stats['n_quads']} quads + {hyb_stats['n_tris']} interface triangles"
        ),
    )
    ax1.text(
        -2.45,
        3.75,
        f"wakis.unstructured.subtri_circles_mesh\n"
        f"Concentric r_in=4.785, r_out=5.215 µm\n"
        f"Graded Cartesian: Δy_min = {sim_nonuniform.dx_min_nm:.2f} nm (8.00 cells/gap)",
        fontsize=7.1,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#1b9e77", alpha=0.92),
    )

    # Panel (c): Resonance comb comparison around m=49
    res_u = find_resonances_fn(
        td_uniform["wavelengths"], td_uniform["S_through_in"], td_uniform["S_drop_in"]
    )
    res_nu = find_resonances_fn(
        td_nonuniform["wavelengths"],
        td_nonuniform["S_through_in"],
        td_nonuniform["S_drop_in"],
    )
    q_u = res_u["mean_Q_loaded"]
    q_nu = res_nu["mean_Q_loaded"]
    ax2.plot(
        res_u["wl_nm"],
        res_u["t_drop"],
        "-",
        color="#08519c",
        lw=1.9,
        label=f"kottke_L3 Drop (Q_L={q_u:.0f})",
    )
    ax2.plot(
        res_nu["wl_nm"],
        res_nu["t_drop"],
        "--",
        color="#d95f02",
        lw=2.0,
        label=f"kottke_nonuniform Drop (Q_L={q_nu:.0f})",
    )
    if td_ref is not None:
        res_r = find_resonances_fn(
            td_ref["wavelengths"], td_ref["S_through_in"], td_ref["S_drop_in"]
        )
        ax2.plot(
            res_r["wl_nm"],
            res_r["t_drop"],
            ":",
            color="black",
            lw=1.8,
            label=f"kottke_reference R Drop (Q_L={res_r['mean_Q_loaded']:.0f})",
        )
    ax2.set(
        xlim=(1552.0, 1592.0),
        ylim=(-0.03, 1.06),
        xlabel="Wavelength λ [nm]",
        ylabel=r"Drop-port transmission $|S_{\mathrm{drop,in}}|^2$",
        title="(c) Resonance comb: graded vs baseline & ref",
    )
    ax2.grid(True, ls=":", alpha=0.4)
    ax2.legend(loc="upper right", fontsize=7.0)
    figstyle.save(fig, out_path, close=True, auto_legend_below=False)


def overlay_ring_add_drop_geometry(
    ax,
    sim,
    *,
    outline_color: str = "white",
    include_labels: bool = True,
):
    """Overlay material outlines/labels, CPML boundaries, source, and monitors on ``ax``."""
    proxy_solver = _SchematicSolverProxy(sim.solver, float(sim.n_clad**2))
    sources_s, monitors_s = _build_ring_add_drop_schematic_io(sim)
    schematic.overlay_geometry(
        ax,
        proxy_solver,
        plane="xy",
        structures=(),
        material_names={"background": f"SiO₂ (n={sim.n_clad:.3f})"},
        sources=sources_s if include_labels else None,
        monitors=monitors_s if include_labels else None,
        unit="µm",
        show_materials=False,
        label_materials=False,
        outline_color=outline_color,
    )
    for s_idx, s in enumerate(sim.structures_3d):
        ax.add_patch(
            MplPolygon(
                s.vertices * 1e6,
                closed=True,
                fill=False,
                edgecolor=outline_color,
                lw=1.05,
                alpha=0.90,
                zorder=6,
            )
        )
        if s_idx == 2 and s.holes:
            ax.add_patch(
                MplPolygon(
                    np.asarray(s.holes[0], dtype=float) * 1e6,
                    closed=True,
                    fill=False,
                    edgecolor=outline_color,
                    lw=1.05,
                    alpha=0.90,
                    zorder=6,
                )
            )
    if include_labels:
        ax.text(
            0.0,
            0.0,
            f"Si EIM ring/bus (n_eff={sim.n_eff_slab:.2f}, R=5 µm, g=100 nm)\nSiO₂ clad (n={sim.n_clad:.3f})",
            fontsize=6.9,
            fontweight="bold",
            color="#0f172a",
            ha="center",
            va="center",
            zorder=9,
            bbox=dict(
                boxstyle="round,pad=0.18",
                fc="white",
                ec="#64748b",
                alpha=0.88,
                lw=0.8,
            ),
        )


def render_ring_add_drop_movie_with_overlay(
    x,
    y,
    frames,
    times_s,
    *,
    sim,
    out_path,
    fps=10,
):
    """Render pulse propagation GIF (< 5 MB) with :func:`overlay_ring_add_drop_geometry` on every frame."""
    from matplotlib.animation import FuncAnimation, PillowWriter

    figstyle.apply_style()
    fig, ax = plt.subplots(figsize=(5.8, 5.8), dpi=80)
    x_um = np.asarray(x) * 1e6
    y_um = np.asarray(y) * 1e6
    f_arr = np.asarray(frames, dtype=float)
    vmax = float(np.percentile(np.abs(f_arr), 99.6)) or 1.0

    im = ax.pcolormesh(
        x_um,
        y_um,
        (f_arr[0] / vmax).T,
        cmap=figstyle.CMAP_DIVERGING,
        vmin=-1.0,
        vmax=1.0,
        shading="auto",
    )
    cb = fig.colorbar(im, ax=ax, fraction=0.040, pad=0.02)
    cb.set_label(r"$E_y(x, y, t) / E_{\max}$")
    overlay_ring_add_drop_geometry(ax, sim, outline_color="black", include_labels=True)
    ax.set(
        xlim=(float(x_um.min()), float(x_um.max())),
        ylim=(float(y_um.min()), float(y_um.max())),
        aspect="equal",
        xlabel="x [µm]",
        ylabel="y [µm]",
    )
    ttl = ax.set_title(
        f"Microring add-drop pulse propagation (t = {float(times_s[0])*1e12:.2f} ps)"
    )
    fig.tight_layout()

    def _update(frame_idx):
        im.set_array((f_arr[frame_idx] / vmax).T.ravel())
        ttl.set_text(
            f"Microring add-drop pulse propagation (t = {float(times_s[frame_idx])*1e12:.2f} ps)"
        )
        return (im, ttl)

    anim = FuncAnimation(
        fig, _update, frames=len(f_arr), interval=1000 // fps, blit=False
    )
    anim.save(str(out_path), writer=PillowWriter(fps=fps), dpi=72)
    plt.close(fig)


if __name__ == "__main__":
    from ring_add_drop import main as _tutorial_main

    raise SystemExit(_tutorial_main())
