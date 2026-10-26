# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Analytical, FIT, and Lugiato–Lefever helper functions for Tutorial 5.11:
Kerr (chi^(3)) Nonlinear Media in FIT and FIT-Informed Soliton Microcombs.

Provides:
1. Cylindrical Bessel/Neumann whispering-gallery eigenmode solver
   (`bessel_ring_mode`, `bessel_ring_dispersion`) for a 2D TM dielectric
   microring resonator, supplying exact analytical benchmarks for azimuthal
   mode frequencies `f_m`, free spectral range `D_1 / (2 pi)`, integrated
   dispersion `D_int(mu) = (1/2) D_2 mu^2`, and radial mode profiles `u_m(r)`.
2. 2D FIT cold-cavity microring dispersion extraction (`run_fit_cold_ring`),
   bus-coupled loaded/intrinsic/external quality factor extraction
   (`run_fit_bus_coupled_ring`), and short 2D FIT Kerr microring resonance-shift
   verification (`run_fit_kerr_ring_shift`) using `wakis.KerrMaterial`,
   `wakis.effective_nonlinear_volume`, and `wakis.kerr_cavity_frequency_shift`.
3. Locally refined non-uniform Cartesian mesh builder (`build_microring_graded_axes`)
   and 3-level uniform resolution convergence study (`run_convergence_study`)
   with second-order Richardson extrapolation.
4. FIT-informed Lugiato–Lefever Equation (LLE) pump-detuning scan
   (`run_lle_detuning_scan`) and `sech^2` bandwidth vs `sqrt(D_2)` scaling
   (`run_lle_dispersion_scaling`) using `wakis.solve_lle`,
   `wakis.lle_cw_bistability`, and `wakis.lle_soliton_analytic`.
5. Setup figure generation (`save_microcomb_setup_figures`) via
   `wakis.schematic.save_setup_figures`, geometry overlay (`overlay_microring_geometry`),
   publication-style figures (`generate_all_figures`), and animated GIF
   (`generate_microcomb_gif`).
"""

from __future__ import annotations

import logging
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle, Rectangle
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c as c_light
from scipy.constants import epsilon_0, hbar, mu_0
from scipy.optimize import brentq
from scipy.special import jv, jvp, yv, yvp

import wakis
from wakis.estimate import estimate, preflight
from wakis.figstyle import (
    CMAP_DIVERGING,
    CMAP_INTENSITY,
    PALETTE,
    apply_style,
    panel_label,
    save,
    spectrum_grid,
)
from wakis.materials import Lorentz
from wakis.nonlinear import (
    KerrMaterial,
    chi3_to_n2,
    effective_nonlinear_volume,
    kerr_cavity_frequency_shift,
    lle_cw_bistability,
    lle_soliton_analytic,
    solve_lle,
)
from wakis.schematic import save_setup_figures
from wakis.sources import CurrentSource, GaussianSource
from wakis.templates.meshing import graded_axis

# ---------------------------------------------------------------------------
# Default physical parameters of the 2D TM microring resonator
# ---------------------------------------------------------------------------

DEFAULT_RING_RADIUS_M = 2.00e-6
DEFAULT_RING_WIDTH_M = 0.60e-6
DEFAULT_BUS_WIDTH_M = 0.28e-6
DEFAULT_GAP_M = 0.16e-6  # 160 nm bus-to-ring gap (>= 4 cells at dx <= 35 nm)
DEFAULT_DOMAIN_HALF_M = 2.80e-6
DEFAULT_SLAB_DZ_M = 0.20e-6
DEFAULT_EPS_INF_ANOM = 4.35
DEFAULT_EPS_INF_NORM = 4.00
DEFAULT_EPS_CLAD = 1.00
DEFAULT_IR_DELTA_EPS = 1.50
DEFAULT_IR_FREQ_HZ = 80.0e12
DEFAULT_CHI3_RING = 2.4e-19  # [m^2/V^2]
DEFAULT_M0 = 17
DEFAULT_M_LIST = (14, 15, 16, 17, 18, 19, 20)


def make_ring_material(
    anomalous: bool = True,
    chi3: float = 0.0,
    method: str = "newton",
) -> KerrMaterial:
    """
    Construct the `KerrMaterial` for the microring core.

    Parameters
    ----------
    anomalous : bool, optional
        If True (default), include the infrared vibrational Lorentz pole at
        ``f_IR = 80 THz`` (``delta_eps = 1.50``) that produces anomalous
        group-velocity dispersion (``D_2 > 0``) around ``m_0 = 17``
        (``f_0 ~ 214.5 THz``). If False, return a dispersionless core
        (``eps_inf = 4.00``) whose geometric whispering-gallery curvature gives
        normal dispersion (``D_2 < 0``).
    chi3 : float, optional
        Third-order susceptibility ``chi^(3)`` [m^2/V^2]. Default ``0.0``.
    method : {'newton', 'fixed_point', 'explicit_ade'}, optional
        Nonlinear update method. Default ``'newton'``.

    Returns
    -------
    KerrMaterial
    """
    if anomalous:
        pole = Lorentz(
            delta_eps=DEFAULT_IR_DELTA_EPS,
            omega_0=2.0 * np.pi * DEFAULT_IR_FREQ_HZ,
            gamma=0.0,
        )
        return KerrMaterial(
            eps_inf=DEFAULT_EPS_INF_ANOM,
            chi3=chi3,
            poles=[pole],
            method=method,
            coupling="edge",
            name="Dispersion-Engineered Kerr Core",
        )
    return KerrMaterial(
        eps_inf=DEFAULT_EPS_INF_NORM,
        chi3=chi3,
        poles=(),
        method=method,
        coupling="edge",
        name="Uncompensated Kerr Core",
    )


def bessel_ring_mode(
    material: KerrMaterial,
    m: int,
    r_arr: np.ndarray | None = None,
    *,
    r_center: float = DEFAULT_RING_RADIUS_M,
    w_ring: float = DEFAULT_RING_WIDTH_M,
    n_clad: float = 1.0,
    dt: float | None = None,
) -> dict:
    """
    Solve the exact 2D cylindrical Bessel/Neumann characteristic equation for the
    fundamental radial ``TM_{m,0}`` whispering-gallery mode of azimuthal order ``m``.

    For out-of-plane ``E_z(r, phi) = u_m(r) cos(m phi)``, continuity of ``E_z``
    and ``H_phi ~ du_m/dr`` at ``R_in = r_center - w_ring / 2`` and
    ``R_out = r_center + w_ring / 2`` yields a closed-form 2x2 determinant in
    terms of Bessel functions ``J_m`` and ``Y_m``.

    Parameters
    ----------
    material : KerrMaterial
        Core material providing ``epsilon(f)`` and ``epsilon_discrete(f, dt)``.
    m : int
        Azimuthal mode order ``m >= 1``.
    r_arr : array_like, optional
        Radial coordinate array [m] on which to evaluate the normalized modal
        profile ``u_m(r)``.
    r_center : float, optional
        Ring centerline radius [m].
    w_ring : float, optional
        Ring radial width [m].
    n_clad : float, optional
        Cladding refractive index [-]. Default ``1.0``.
    dt : float, optional
        If provided, use ``material.epsilon_discrete(f, dt)`` to match the
        discrete ADE pole dispersion at timestep ``dt`` [s].

    Returns
    -------
    dict
        Keys ``'m'``, ``'frequency_hz'``, ``'omega_rad_s'``, ``'n_core'``,
        and ``'profile'`` (if ``r_arr`` is given).
    """
    r_in = float(r_center) - 0.5 * float(w_ring)
    r_out = float(r_center) + 0.5 * float(w_ring)
    n1 = float(n_clad)

    def _eps_at(freq):
        if dt is not None:
            return float(np.real(material.epsilon_discrete(freq, dt)))
        return float(np.real(material.epsilon(freq)))

    def _char_det(freq):
        n2 = np.sqrt(max(_eps_at(freq), 1.01))
        k0 = 2.0 * np.pi * freq / c_light
        k1, k2 = n1 * k0, n2 * k0
        j1_i, dj1_i = jv(m, k1 * r_in), k1 * jvp(m, k1 * r_in)
        j2_i, dj2_i = jv(m, k2 * r_in), k2 * jvp(m, k2 * r_in)
        y2_i, dy2_i = yv(m, k2 * r_in), k2 * yvp(m, k2 * r_in)
        b_coef = j1_i * dy2_i - dj1_i * y2_i
        c_coef = dj1_i * j2_i - j1_i * dj2_i
        u_o = b_coef * jv(m, k2 * r_out) + c_coef * yv(m, k2 * r_out)
        du_o = b_coef * k2 * jvp(m, k2 * r_out) + c_coef * k2 * yvp(m, k2 * r_out)
        return float(u_o * k1 * yvp(m, k1 * r_out) - du_o * yv(m, k1 * r_out))

    f_grid = np.linspace(130.0e12, 265.0e12, 550)
    vals = [_char_det(ff) for ff in f_grid]
    f_root = None
    for k in range(len(f_grid) - 1):
        if vals[k] * vals[k + 1] <= 0.0:
            f_root = float(brentq(_char_det, f_grid[k], f_grid[k + 1], rtol=1e-12))
            break
    if f_root is None:
        raise RuntimeError(f"Could not bracket Bessel WGM root for m={m}.")

    n2_root = float(np.sqrt(_eps_at(f_root)))
    out = {
        "m": int(m),
        "frequency_hz": f_root,
        "omega_rad_s": float(2.0 * np.pi * f_root),
        "n_core": n2_root,
    }
    if r_arr is not None:
        k0 = 2.0 * np.pi * f_root / c_light
        k1, k2 = n1 * k0, n2_root * k0
        j1_i, dj1_i = jv(m, k1 * r_in), k1 * jvp(m, k1 * r_in)
        j2_i, dj2_i = jv(m, k2 * r_in), k2 * jvp(m, k2 * r_in)
        y2_i, dy2_i = yv(m, k2 * r_in), k2 * yvp(m, k2 * r_in)
        b_coef = j1_i * dy2_i - dj1_i * y2_i
        c_coef = dj1_i * j2_i - j1_i * dj2_i
        u_o = b_coef * jv(m, k2 * r_out) + c_coef * yv(m, k2 * r_out)
        d_coef = u_o / yv(m, k1 * r_out)

        r = np.maximum(np.asarray(r_arr, dtype=float), 1e-12)
        u = np.empty_like(r)
        mask_in = r < r_in
        mask_core = np.logical_and(r >= r_in, r <= r_out)
        mask_out = r > r_out
        u[mask_in] = jv(m, k1 * r[mask_in])
        u[mask_core] = b_coef * jv(m, k2 * r[mask_core]) + c_coef * yv(
            m, k2 * r[mask_core]
        )
        u[mask_out] = d_coef * yv(m, k1 * r[mask_out])
        u[mask_out] *= np.exp(-((np.maximum(r[mask_out] - r_out, 0.0) / 0.35e-6) ** 2))
        u /= float(np.max(np.abs(u)))
        out["profile"] = u
    return out


def bessel_ring_dispersion(
    material: KerrMaterial,
    m_list: tuple[int, ...] = DEFAULT_M_LIST,
    m0: int = DEFAULT_M0,
    *,
    r_center: float = DEFAULT_RING_RADIUS_M,
    w_ring: float = DEFAULT_RING_WIDTH_M,
    dt: float | None = None,
) -> dict:
    """
    Compute the analytical azimuthal resonance family ``omega_mu`` and integrated
    dispersion ``D_int(mu) = omega_mu - omega_0 - D_1 mu`` across ``m_list``.
    """
    m_arr = np.asarray(m_list, dtype=int)
    mu_arr = m_arr - int(m0)
    freqs = np.array(
        [
            bessel_ring_mode(
                material, int(m), r_center=r_center, w_ring=w_ring, dt=dt
            )["frequency_hz"]
            for m in m_arr
        ],
        dtype=float,
    )
    p2 = np.polyfit(mu_arr, freqs, 2)
    d2_hz = float(2.0 * p2[0])
    d1_hz = float(p2[1])
    f0_fit = float(p2[2])
    idx0 = int(np.where(mu_arr == 0)[0][0])
    f0_exact = float(freqs[idx0])
    d_int_hz = freqs - f0_exact - d1_hz * mu_arr
    return {
        "m_list": m_arr.tolist(),
        "mu": mu_arr.tolist(),
        "frequencies_hz": freqs.tolist(),
        "f0_hz": f0_exact,
        "f0_poly_hz": f0_fit,
        "D1_hz": d1_hz,
        "D2_hz": d2_hz,
        "D1_rad_s": float(2.0 * np.pi * d1_hz),
        "D2_rad_s": float(2.0 * np.pi * d2_hz),
        "D_int_hz": d_int_hz.tolist(),
        "D_int_rad_s": (2.0 * np.pi * d_int_hz).tolist(),
    }


def build_microring_graded_axes(
    dx_fine: float = 20.0e-9,
    dx_coarse: float = 35.0e-9,
    r_center: float = DEFAULT_RING_RADIUS_M,
    w_ring: float = DEFAULT_RING_WIDTH_M,
    gap: float = DEFAULT_GAP_M,
    w_bus: float = DEFAULT_BUS_WIDTH_M,
    domain_half: float = DEFAULT_DOMAIN_HALF_M,
    dx_mid: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build locally refined non-uniform ``(x_nodes, y_nodes)`` Cartesian grid axes
    that place ``dx_fine`` across the ring waveguide walls and the bus-to-ring
    coupling gap, ``dx_mid`` across the central span (maintaining >= 23 cells/lambda
    everywhere inside the ring core), and grade smoothly to ``dx_coarse`` in the
    outer cladding and PML.
    """
    if dx_mid is None:
        dx_mid = min(float(dx_coarse), 1.175 * float(dx_fine))
    r_out = r_center + 0.5 * w_ring
    r_in = r_center - 0.5 * w_ring
    y_bus_top = r_out + gap + w_bus
    x_points = [
        -domain_half,
        -r_out - 0.10e-6,
        -r_in + 0.25e-6,
        r_in - 0.25e-6,
        r_out + 0.10e-6,
        domain_half,
    ]
    x_sizes = [dx_coarse, dx_fine, dx_mid, dx_fine, dx_coarse]
    x_nodes = graded_axis(x_points, x_sizes, min_cells=4, grading=1.18)

    y_top_fine = min(y_bus_top + 0.06e-6, domain_half - 0.12e-6)
    y_points = [
        -domain_half,
        -r_out - 0.10e-6,
        -r_in + 0.25e-6,
        r_in - 0.25e-6,
        y_top_fine,
        domain_half,
    ]
    y_sizes = [dx_coarse, dx_fine, dx_mid, dx_fine, dx_coarse]
    y_nodes = graded_axis(y_points, y_sizes, min_cells=4, grading=1.18)
    return x_nodes, y_nodes


def _dual_cell_geometry_2d(
    x_nodes: np.ndarray,
    y_nodes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return 2D meshgrids ``(xc_d, yc_d, dx_d, dy_d)`` of the dual-cell centres
    and widths surrounding each primary ``E_z`` node ``(x_nodes[i], y_nodes[j])``.
    """
    x_e = np.asarray(x_nodes, dtype=float)
    y_e = np.asarray(y_nodes, dtype=float)
    x_lo = np.empty_like(x_e)
    x_hi = np.empty_like(x_e)
    x_lo[1:] = 0.5 * (x_e[:-1] + x_e[1:])
    x_lo[0] = x_e[0] - 0.5 * (x_e[1] - x_e[0])
    x_hi[:-1] = 0.5 * (x_e[:-1] + x_e[1:])
    x_hi[-1] = x_e[-1] + 0.5 * (x_e[-1] - x_e[-2])

    y_lo = np.empty_like(y_e)
    y_hi = np.empty_like(y_e)
    y_lo[1:] = 0.5 * (y_e[:-1] + y_e[1:])
    y_lo[0] = y_e[0] - 0.5 * (y_e[1] - y_e[0])
    y_hi[:-1] = 0.5 * (y_e[:-1] + y_e[1:])
    y_hi[-1] = y_e[-1] + 0.5 * (y_e[-1] - y_e[-2])

    xc_d, yc_d = np.meshgrid(0.5 * (x_lo + x_hi), 0.5 * (y_lo + y_hi), indexing="ij")
    dx_d, dy_d = np.meshgrid(x_hi - x_lo, y_hi - y_lo, indexing="ij")
    return xc_d, yc_d, dx_d, dy_d


def _build_ring_fill_fraction(
    x_nodes: np.ndarray,
    y_nodes: np.ndarray,
    *,
    r_center: float = DEFAULT_RING_RADIUS_M,
    w_ring: float = DEFAULT_RING_WIDTH_M,
    include_bus: bool = False,
    gap: float = DEFAULT_GAP_M,
    w_bus: float = DEFAULT_BUS_WIDTH_M,
    n_sub: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute ``n_sub x n_sub`` subpixel fill fractions ``(fill_ring, fill_bus)``
    over the exact dual cell around each ``(Nx, Ny)`` ``E_z`` node.
    """
    nx = len(x_nodes)
    ny = len(y_nodes)
    xc_d, yc_d, dx_m, dy_m = _dual_cell_geometry_2d(x_nodes, y_nodes)

    r_in = r_center - 0.5 * w_ring
    r_out = r_center + 0.5 * w_ring
    y_bus_lo = r_out + gap
    y_bus_hi = y_bus_lo + w_bus

    sub = (np.arange(int(n_sub), dtype=float) + 0.5) / float(n_sub) - 0.5
    fill_ring = np.zeros((nx, ny), dtype=float)
    fill_bus = np.zeros((nx, ny), dtype=float)
    for sx in sub:
        for sy in sub:
            xs = xc_d + sx * dx_m
            ys = yc_d + sy * dy_m
            rr = np.hypot(xs, ys)
            fill_ring += np.logical_and(rr >= r_in, rr <= r_out).astype(float)
            if include_bus:
                fill_bus += np.logical_and(ys >= y_bus_lo, ys <= y_bus_hi).astype(float)
    inv_sub2 = 1.0 / float(n_sub * n_sub)
    fill_ring *= inv_sub2
    fill_bus *= inv_sub2
    return fill_ring, fill_bus


def _attach_2d_tm_material(
    solver: wakis.SolverFIT3D,
    material: KerrMaterial,
    fill_2d: np.ndarray,
) -> None:
    """Attach a `KerrMaterial` to all `E_z` edges in `solver` with 2D fill weights `fill_2d`."""
    ii, jj = np.nonzero(fill_2d > 0.0)
    w_ij = fill_2d[ii, jj]
    idx_list = []
    w_list = []
    for k in range(solver.Nz):
        idx_k = ii + jj * solver.Nx + k * solver.Nx * solver.Ny + 2 * solver.N
        idx_list.append(idx_k)
        w_list.append(w_ij)
    solver._add_nonlinear_edges(
        material, np.concatenate(idx_list), np.concatenate(w_list), eps_bg=DEFAULT_EPS_CLAD
    )


def run_fit_cold_ring(
    dx_m: float = 20.0e-9,
    *,
    anomalous: bool = True,
    graded: bool = False,
    n_steps: int | None = None,
    m_list: tuple[int, ...] = DEFAULT_M_LIST,
    m0: int = DEFAULT_M0,
) -> dict:
    """
    Run a 2D FIT cold-cavity microring simulation and extract the azimuthal
    resonance frequencies `f_m` and integrated dispersion `D_int(mu)` via
    2D core modal projection + `wakis.harminv`.
    """
    t0 = time.perf_counter()
    z_nodes = np.linspace(-0.5 * DEFAULT_SLAB_DZ_M, 0.5 * DEFAULT_SLAB_DZ_M, 2)
    if graded:
        x_nodes, y_nodes = build_microring_graded_axes(dx_fine=dx_m)
        grid = wakis.GridFIT3D(
            -DEFAULT_DOMAIN_HALF_M,
            DEFAULT_DOMAIN_HALF_M,
            -DEFAULT_DOMAIN_HALF_M,
            DEFAULT_DOMAIN_HALF_M,
            -0.5 * DEFAULT_SLAB_DZ_M,
            0.5 * DEFAULT_SLAB_DZ_M,
            len(x_nodes) - 1,
            len(y_nodes) - 1,
            1,
            x=x_nodes,
            y=y_nodes,
            z=z_nodes,
            verbose=0,
        )
    else:
        n_xy = int(round(2.0 * DEFAULT_DOMAIN_HALF_M / dx_m))
        if n_xy % 2 == 1:
            n_xy += 1
        grid = wakis.GridFIT3D(
            -DEFAULT_DOMAIN_HALF_M,
            DEFAULT_DOMAIN_HALF_M,
            -DEFAULT_DOMAIN_HALF_M,
            DEFAULT_DOMAIN_HALF_M,
            -0.5 * DEFAULT_SLAB_DZ_M,
            0.5 * DEFAULT_SLAB_DZ_M,
            n_xy,
            n_xy,
            1,
            z=z_nodes,
            verbose=0,
        )

    solver = wakis.SolverFIT3D(
        grid,
        cfln=0.5,
        bc_low=["cpml", "cpml", "periodic"],
        bc_high=["cpml", "cpml", "periodic"],
        n_pml=10,
        bg=[DEFAULT_EPS_CLAD, 1.0, 0.0],
        verbose=0,
    )
    if n_steps is None:
        n_steps = int(round(150.0e-15 / solver.dt))

    est = estimate((grid.Nx, grid.Ny, grid.Nz), steps=n_steps, cpml=True)
    preflight(est, "laptop")

    mat = make_ring_material(anomalous=anomalous, chi3=0.0)
    x_e = solver.grid.x[: solver.Nx]
    y_e = solver.grid.y[: solver.Ny]
    fill_ring, _ = _build_ring_fill_fraction(x_e, y_e, include_bus=False, n_sub=16)
    _attach_2d_tm_material(solver, mat, fill_ring)

    xx, yy = np.meshgrid(x_e, y_e, indexing="ij")
    rr = np.hypot(xx, yy)
    phi = np.arctan2(yy, xx)
    _, _, dx_d, dy_d = _dual_cell_geometry_2d(x_e, y_e)
    dv_2d = (dx_d * dy_d) * DEFAULT_SLAB_DZ_M

    r_in = DEFAULT_RING_RADIUS_M - 0.5 * DEFAULT_RING_WIDTH_M
    r_out = DEFAULT_RING_RADIUS_M + 0.5 * DEFAULT_RING_WIDTH_M
    core_mask = np.logical_and(rr >= r_in, rr <= r_out)
    ix_c, iy_c = np.nonzero(core_mask)
    flat_c = ix_c + iy_c * solver.Nx + 2 * solver.N
    sqrt_dv_c = np.sqrt(dv_2d[ix_c, iy_c])

    bessel_ref = bessel_ring_dispersion(mat, m_list=m_list, m0=m0, dt=solver.dt)
    ez_init = np.zeros_like(xx)
    mode_m0_profile = None
    basis_cols = []
    for m in m_list:
        bm = bessel_ring_mode(mat, int(m), rr, dt=solver.dt)
        mode_m = bm["profile"] * np.cos(int(m) * phi)
        ez_init += mode_m
        basis_cols.append(mode_m[ix_c, iy_c] * sqrt_dv_c)
        if int(m) == int(m0):
            mode_m0_profile = mode_m
    for k in range(solver.Nz):
        solver.E[: solver.Nx, : solver.Ny, k, "z"] = 1.0e5 * ez_init

    a_harm = np.column_stack(basis_cols)
    proj_pinv = np.linalg.pinv(a_harm) * sqrt_dv_c[None, :]
    dx_min = float(np.min(np.diff(x_e)))

    traces = np.zeros((len(m_list), n_steps), dtype=float)
    snap_mid = None
    for n in range(n_steps):
        solver.one_step()
        traces[:, n] = proj_pinv @ solver.E.array[flat_c]
        if n == n_steps // 2:
            snap_mid = solver.E[: solver.Nx, : solver.Ny, 0, "z"].copy()

    stride = max(1, n_steps // 700)
    f_fit = []
    q_fit = []
    for idx_m, m in enumerate(m_list):
        f_est = bessel_ref["frequencies_hz"][idx_m]
        res = wakis.harminv(
            traces[idx_m, 150::stride],
            solver.dt * stride,
            fmin=f_est - 5.0e12,
            fmax=f_est + 5.0e12,
        )
        dom = int(np.argmax(res.amplitude))
        f_fit.append(float(res.frequency[dom]))
        q_fit.append(float(abs(res.Q[dom])))

    f_fit_arr = np.asarray(f_fit, dtype=float)
    mu_arr = np.asarray(m_list, dtype=int) - int(m0)
    p2 = np.polyfit(mu_arr, f_fit_arr, 2)
    d2_fit_hz = float(2.0 * p2[0])
    d1_fit_hz = float(p2[1])
    idx0 = int(np.where(mu_arr == 0)[0][0])
    f0_fit_hz = float(f_fit_arr[idx0])
    d_int_fit_hz = f_fit_arr - f0_fit_hz - d1_fit_hz * mu_arr

    eps_core_f0 = float(np.real(mat.epsilon_discrete(f0_fit_hz, solver.dt)))
    eps_2d = DEFAULT_EPS_CLAD + fill_ring * (eps_core_f0 - DEFAULT_EPS_CLAD)
    v_eff_standing = float(
        effective_nonlinear_volume(
            mode_m0_profile**2, dV=dv_2d, eps_r=eps_2d, nl_mask=fill_ring
        )
    )
    # Travelling-wave WGM has uniform azimuthal intensity |exp(-i m0 phi)|^2 = 1
    # (1.5x larger mode volume than cos(m0 phi) standing wave)
    bm0_rad = bessel_ring_mode(mat, int(m0), rr, dt=solver.dt)["profile"]
    v_eff_travelling = float(
        effective_nonlinear_volume(
            bm0_rad**2, dV=dv_2d, eps_r=eps_2d, nl_mask=fill_ring
        )
    )

    n_core_f0 = float(np.sqrt(eps_core_f0))
    lam_core_m = c_light / (f0_fit_hz * n_core_f0)
    ppw_core = lam_core_m / dx_min
    cells_across_gap = DEFAULT_GAP_M / dx_min

    return {
        "anomalous": bool(anomalous),
        "graded": bool(graded),
        "dx_min_m": dx_min,
        "dx_max_m": float(np.max(np.diff(x_e))),
        "nx": int(solver.Nx),
        "ny": int(solver.Ny),
        "total_cells_2d": int(solver.Nx * solver.Ny),
        "dt_s": float(solver.dt),
        "n_steps": int(n_steps),
        "ppw_core": float(ppw_core),
        "cells_across_gap": float(cells_across_gap),
        "m_list": list(m_list),
        "mu": mu_arr.tolist(),
        "f_fit_hz": f_fit_arr.tolist(),
        "f_bessel_hz": bessel_ref["frequencies_hz"],
        "q_cold": q_fit,
        "f0_fit_hz": f0_fit_hz,
        "f0_bessel_hz": float(bessel_ref["f0_hz"]),
        "D1_fit_hz": d1_fit_hz,
        "D1_bessel_hz": float(bessel_ref["D1_hz"]),
        "D2_fit_hz": d2_fit_hz,
        "D2_bessel_hz": float(bessel_ref["D2_hz"]),
        "D_int_fit_hz": d_int_fit_hz.tolist(),
        "D_int_bessel_hz": bessel_ref["D_int_hz"],
        "f_rel_errors": (
            np.abs(f_fit_arr - np.array(bessel_ref["frequencies_hz"]))
            / np.array(bessel_ref["frequencies_hz"])
        ).tolist(),
        "D1_rel_error": abs(d1_fit_hz - bessel_ref["D1_hz"]) / abs(bessel_ref["D1_hz"]),
        "D2_rel_error": abs(d2_fit_hz - bessel_ref["D2_hz"]) / abs(bessel_ref["D2_hz"]),
        "eps_core_f0": eps_core_f0,
        "n_core_f0": n_core_f0,
        "V_eff_standing_m3": v_eff_standing,
        "V_eff_travelling_m3": v_eff_travelling,
        "x_e": x_e,
        "y_e": y_e,
        "fill_ring": fill_ring,
        "snap_ez": snap_mid,
        "mode_m0_profile": mode_m0_profile,
        "radial_m0_profile": bm0_rad,
        "dv_2d": dv_2d,
        "eps_2d": eps_2d,
        "wall_time_s": float(time.perf_counter() - t0),
    }


def run_fit_bus_coupled_ring(
    dx_m: float = 25.0e-9,
    gap_m: float = DEFAULT_GAP_M,
    n_steps: int = 3600,
    m0: int = DEFAULT_M0,
) -> dict:
    """
    Run a 2D FIT bus-coupled microring simulation at gap `gap_m` and extract
    the loaded quality factor `Q_L`, intrinsic `Q_0`, and external coupling
    `Q_ex` (with `1 / Q_L = 1 / Q_0 + 1 / Q_ex`) of mode `m0`.
    """
    t0 = time.perf_counter()
    n_xy = int(round(2.0 * DEFAULT_DOMAIN_HALF_M / dx_m))
    if n_xy % 2 == 1:
        n_xy += 1
    z_nodes = np.linspace(-0.5 * DEFAULT_SLAB_DZ_M, 0.5 * DEFAULT_SLAB_DZ_M, 2)
    grid = wakis.GridFIT3D(
        -DEFAULT_DOMAIN_HALF_M,
        DEFAULT_DOMAIN_HALF_M,
        -DEFAULT_DOMAIN_HALF_M,
        DEFAULT_DOMAIN_HALF_M,
        -0.5 * DEFAULT_SLAB_DZ_M,
        0.5 * DEFAULT_SLAB_DZ_M,
        n_xy,
        n_xy,
        1,
        z=z_nodes,
        verbose=0,
    )
    est = estimate((grid.Nx, grid.Ny, grid.Nz), steps=n_steps, cpml=True)
    preflight(est, "laptop")

    def _run_q(include_bus: bool):
        solver = wakis.SolverFIT3D(
            grid,
            cfln=0.5,
            bc_low=["cpml", "cpml", "periodic"],
            bc_high=["cpml", "cpml", "periodic"],
            n_pml=10,
            bg=[DEFAULT_EPS_CLAD, 1.0, 0.0],
            verbose=0,
        )
        mat = make_ring_material(anomalous=True, chi3=0.0)
        x_e = solver.grid.x[: solver.Nx]
        y_e = solver.grid.y[: solver.Ny]
        fill_ring, fill_bus = _build_ring_fill_fraction(
            x_e, y_e, include_bus=include_bus, gap=gap_m, n_sub=16
        )
        fill_tot = np.clip(fill_ring + fill_bus, 0.0, 1.0)
        _attach_2d_tm_material(solver, mat, fill_tot)

        xx, yy = np.meshgrid(x_e, y_e, indexing="ij")
        rr = np.hypot(xx, yy)
        phi = np.arctan2(yy, xx)
        bm = bessel_ring_mode(mat, int(m0), rr, dt=solver.dt)
        ez_init = bm["profile"] * np.cos(int(m0) * phi)
        for k in range(solver.Nz):
            solver.E[: solver.Nx, : solver.Ny, k, "z"] = 1.0e5 * ez_init

        r_probe = DEFAULT_RING_RADIUS_M + 0.05e-6
        ring_mask = np.abs(rr - r_probe) <= 1.25 * dx_m
        ix_p, iy_p = np.nonzero(ring_mask)
        phi_p = phi[ix_p, iy_p]
        cos_vec = np.cos(int(m0) * phi_p)
        cos_vec /= float(np.sum(cos_vec**2))
        flat_p = ix_p + iy_p * solver.Nx + 2 * solver.N

        trace = np.zeros(n_steps, dtype=float)
        snap = None
        for n in range(n_steps):
            solver.one_step()
            trace[n] = float(np.dot(cos_vec, solver.E.array[flat_p]))
            if n == int(0.65 * n_steps):
                snap = solver.E[: solver.Nx, : solver.Ny, 0, "z"].copy()

        stride = 6
        f_est = bm["frequency_hz"]
        res = wakis.harminv(
            trace[150::stride],
            solver.dt * stride,
            fmin=f_est - 6.0e12,
            fmax=f_est + 6.0e12,
        )
        dom = int(np.argmax(res.amplitude))
        return (
            float(res.frequency[dom]),
            float(abs(res.Q[dom])),
            snap,
            fill_tot,
            x_e,
            y_e,
        )

    f_cold, q_0, _, _, _, _ = _run_q(include_bus=False)
    f_bus, q_loaded, snap_bus, fill_bus_tot, x_e, y_e = _run_q(include_bus=True)

    # 1 / Q_ex = 1 / Q_L - 1 / Q_0
    inv_q_ex = max(1.0 / q_loaded - 1.0 / q_0, 1.0 / (5.0 * q_0))
    q_ex = float(1.0 / inv_q_ex)
    kappa_0 = float(2.0 * np.pi * f_bus / q_0)
    kappa_ex = float(2.0 * np.pi * f_bus / q_ex)
    kappa_tot = float(kappa_0 + kappa_ex)

    return {
        "gap_m": float(gap_m),
        "cells_across_gap": float(gap_m / dx_m),
        "f_cold_hz": f_cold,
        "f_bus_hz": f_bus,
        "Q_0": q_0,
        "Q_loaded": q_loaded,
        "Q_ex": q_ex,
        "kappa_0_rad_s": kappa_0,
        "kappa_ex_rad_s": kappa_ex,
        "kappa_tot_rad_s": kappa_tot,
        "kappa_tot_ghz": float(kappa_tot / (2.0 * np.pi * 1e9)),
        "eta_coupling": float(kappa_ex / kappa_tot),
        "snap_bus_ez": snap_bus,
        "fill_bus_tot": fill_bus_tot,
        "x_e": x_e,
        "y_e": y_e,
        "wall_time_s": float(time.perf_counter() - t0),
    }


def run_fit_kerr_ring_shift(
    dx_m: float = 30.0e-9,
    chi3: float = DEFAULT_CHI3_RING,
    E_peaks: tuple[float, ...] = (3.0e7, 6.0e7, 9.0e7),
    n_steps: int = 2400,
    m0: int = DEFAULT_M0,
) -> dict:
    """
    Run a short 2D FIT Kerr simulation of the microring (`KerrMaterial` with
    `chi3 > 0`) at multiple modal field amplitudes `E_peaks` and verify the
    nonlinear resonance shift `Delta f_NL` against perturbation theory with the
    mode's effective nonlinear volume `V_eff,NL`.
    """
    t0 = time.perf_counter()
    n_xy = int(round(2.0 * DEFAULT_DOMAIN_HALF_M / dx_m))
    if n_xy % 2 == 1:
        n_xy += 1
    z_nodes = np.linspace(-0.5 * DEFAULT_SLAB_DZ_M, 0.5 * DEFAULT_SLAB_DZ_M, 2)
    grid = wakis.GridFIT3D(
        -DEFAULT_DOMAIN_HALF_M,
        DEFAULT_DOMAIN_HALF_M,
        -DEFAULT_DOMAIN_HALF_M,
        DEFAULT_DOMAIN_HALF_M,
        -0.5 * DEFAULT_SLAB_DZ_M,
        0.5 * DEFAULT_SLAB_DZ_M,
        n_xy,
        n_xy,
        1,
        z=z_nodes,
        verbose=0,
    )
    est = estimate((grid.Nx, grid.Ny, grid.Nz), steps=n_steps, cpml=True)
    preflight(est, "laptop")

    x_e = grid.x[: grid.Nx]
    y_e = grid.y[: grid.Ny]
    fill_ring, _ = _build_ring_fill_fraction(x_e, y_e, include_bus=False, n_sub=16)
    xx, yy = np.meshgrid(x_e, y_e, indexing="ij")
    rr = np.hypot(xx, yy)
    phi = np.arctan2(yy, xx)
    _, _, dx_d, dy_d = _dual_cell_geometry_2d(x_e, y_e)
    dv_cell = (dx_d * dy_d) * DEFAULT_SLAB_DZ_M

    mat_lin = make_ring_material(anomalous=False, chi3=0.0)
    dt_est = 0.5 / (c_light * np.sqrt(2.0 / dx_m**2 + 1.0 / DEFAULT_SLAB_DZ_M**2))
    bm0 = bessel_ring_mode(mat_lin, int(m0), rr, dt=dt_est)
    mode_2d = bm0["profile"] * np.cos(int(m0) * phi)
    mode_2d /= float(np.max(np.abs(mode_2d)))

    r_probe = DEFAULT_RING_RADIUS_M + 0.05e-6
    ring_mask = np.abs(rr - r_probe) <= 1.25 * dx_m
    ix_p, iy_p = np.nonzero(ring_mask)
    phi_p = phi[ix_p, iy_p]
    cos_vec = np.cos(int(m0) * phi_p)
    cos_vec /= float(np.sum(cos_vec**2))

    def _run_single(chi3_val: float, e_amp: float) -> float:
        solver = wakis.SolverFIT3D(
            grid,
            cfln=0.5,
            bc_low=["cpml", "cpml", "periodic"],
            bc_high=["cpml", "cpml", "periodic"],
            n_pml=10,
            bg=[DEFAULT_EPS_CLAD, 1.0, 0.0],
            verbose=0,
        )
        flat_p = ix_p + iy_p * solver.Nx + 2 * solver.N
        mat = make_ring_material(anomalous=False, chi3=chi3_val)
        _attach_2d_tm_material(solver, mat, fill_ring)
        for k in range(solver.Nz):
            solver.E[: solver.Nx, : solver.Ny, k, "z"] = e_amp * mode_2d

        trace = np.zeros(n_steps, dtype=float)
        for n in range(n_steps):
            solver.one_step()
            trace[n] = float(np.dot(cos_vec, solver.E.array[flat_p]))

        stride = 6
        f_est = bm0["frequency_hz"]
        res = wakis.harminv(
            trace[100::stride],
            solver.dt * stride,
            fmin=f_est - 8.0e12,
            fmax=f_est + 8.0e12,
        )
        dom = int(np.argmax(res.amplitude))
        return float(res.frequency[dom])

    f_lin = _run_single(0.0, E_peaks[0])
    eps_2d = DEFAULT_EPS_CLAD + fill_ring * (DEFAULT_EPS_INF_NORM - DEFAULT_EPS_CLAD)
    v_eff_nl = float(
        effective_nonlinear_volume(
            mode_2d**2, dV=dv_cell, eps_r=eps_2d, nl_mask=fill_ring
        )
    )

    df_fit_hz = []
    df_an_hz = []
    stored_u_j = []
    rel_errors = []
    for e_amp in E_peaks:
        f_nl = _run_single(chi3, float(e_amp))
        df_m = f_nl - f_lin
        shift_info = kerr_cavity_frequency_shift(
            2.0 * np.pi * f_lin,
            chi3,
            e_field=float(e_amp) * mode_2d,
            dV=dv_cell,
            eps_r=eps_2d,
            nl_mask=fill_ring,
        )
        df_a = float(shift_info["delta_f"])
        df_fit_hz.append(df_m)
        df_an_hz.append(df_a)
        stored_u_j.append(float(shift_info["stored_energy"]))
        rel_errors.append(abs(df_m - df_a) / abs(df_a))

    # Single-photon Kerr shift g_0 = hbar * omega_0^2 * c * n2 / (n0^2 * V_eff,NL)
    n0 = float(np.sqrt(DEFAULT_EPS_INF_NORM))
    n2 = float(chi3_to_n2(chi3, eps_r=DEFAULT_EPS_INF_NORM))
    omega_0 = 2.0 * np.pi * f_lin
    g_nl_per_joule = float((omega_0 * c_light * n2) / (n0**2 * v_eff_nl))
    g0_single_photon_rad_s = float(hbar * omega_0 * g_nl_per_joule)

    return {
        "dx_m": float(dx_m),
        "chi3_m2_v2": float(chi3),
        "n2_m2_w": n2,
        "f_lin_hz": f_lin,
        "V_eff_nl_m3": v_eff_nl,
        "V_eff_nl_um3": float(v_eff_nl * 1e18),
        "g_nl_rad_s_per_j": g_nl_per_joule,
        "g0_single_photon_hz": float(g0_single_photon_rad_s / (2.0 * np.pi)),
        "E_peaks_v_m": [float(x) for x in E_peaks],
        "stored_energy_j": stored_u_j,
        "df_fit_hz": df_fit_hz,
        "df_analytic_hz": df_an_hz,
        "rel_errors": rel_errors,
        "max_rel_error": float(np.max(rel_errors)),
        "wall_time_s": float(time.perf_counter() - t0),
    }


def _compute_observed_order_and_richardson(
    dx_levels_m: Sequence[float],
    values: Sequence[float],
) -> tuple[float, float]:
    """
    Compute the observed 3-grid convergence order ``p_obs`` and corresponding
    Richardson extrapolation for a potentially non-uniform grid-refinement
    triplet ``h_0 > h_1 > h_2`` by solving
    ``(h_0^p - h_1^p) / (h_1^p - h_2^p) = (y_1 - y_0) / (y_2 - y_1)``.
    """
    h = np.asarray(dx_levels_m, dtype=float)
    y = np.asarray(values, dtype=float)
    if h.size < 3 or abs(y[-1] - y[-2]) < 1e-30:
        r21 = float(h[-2] / h[-1])
        return 2.0, float(y[-1] + (y[-1] - y[-2]) / (r21**2 - 1.0))
    ratio_y = float((y[-2] - y[-3]) / (y[-1] - y[-2]))
    if ratio_y <= 1.0:
        p_obs = 2.0
    else:
        try:
            p_obs = float(
                brentq(
                    lambda p: ((h[-3] ** p - h[-2] ** p) / (h[-2] ** p - h[-1] ** p))
                    - ratio_y,
                    0.05,
                    20.0,
                )
            )
        except ValueError:
            p_obs = 2.0
    r21_p = float((h[-2] / h[-1]) ** p_obs)
    y_rich_obs = float(y[-1] + (y[-1] - y[-2]) / (r21_p - 1.0))
    return p_obs, y_rich_obs


def augment_convergence_dict(conv_res: dict) -> dict:
    """
    Ensure ``conv_res`` contains both nominal ``p = 2`` and observed-order
    ``p_obs`` Richardson extrapolation fields and the resulting ``D_2`` band.
    """
    out = dict(conv_res)
    dx_m = [float(x) * 1e-9 for x in out["dx_nm"]]
    p_f0, f0_obs = _compute_observed_order_and_richardson(dx_m, out["f0_hz"])
    p_d1, d1_obs = _compute_observed_order_and_richardson(dx_m, out["D1_hz"])
    p_d2, d2_obs = _compute_observed_order_and_richardson(dx_m, out["D2_hz"])
    d2_bessel = float(out["D2_bessel_hz"])
    d2_p2 = float(out["D2_richardson_hz"])
    d2_obs_err = abs(d2_obs - d2_bessel) / abs(d2_bessel)
    d2_p2_err = abs(d2_p2 - d2_bessel) / abs(d2_bessel)
    out["f0_observed_order"] = p_f0
    out["f0_richardson_obs_hz"] = f0_obs
    out["D1_observed_order"] = p_d1
    out["D1_richardson_obs_hz"] = d1_obs
    out["D2_observed_order"] = p_d2
    out["D2_richardson_obs_hz"] = d2_obs
    out["D2_richardson_obs_rel_err"] = d2_obs_err
    out["D2_richardson_band_hz"] = [min(d2_obs, d2_p2), max(d2_obs, d2_p2)]
    out["D2_richardson_band_rel_err"] = [min(d2_p2_err, d2_obs_err), max(d2_p2_err, d2_obs_err)]
    return out


def run_convergence_study(
    dx_levels_m: tuple[float, ...] = (30.0e-9, 25.0e-9, 20.0e-9),
    n_steps: int | None = None,
) -> dict:
    """
    Run a 3-level uniform resolution convergence study (`Coarse`, `Medium`,
    `Fine`, all satisfying `>= 20 cells/lambda` in the core and `>= 4 cells`
    across the 160 nm coupling gap) plus a locally refined graded mesh, and
    compute both nominal second-order (`p = 2`) and observed-order (`p_obs`)
    Richardson extrapolations.
    """
    t0 = time.perf_counter()
    level_names = ["Coarse", "Medium", "Fine"][: len(dx_levels_m)]
    f0_list = []
    d1_list = []
    d2_list = []
    veff_list = []
    ppw_list = []
    gap_cells_list = []
    cells_list = []
    times_list = []

    for dx in dx_levels_m:
        res = run_fit_cold_ring(dx_m=float(dx), anomalous=True, graded=False, n_steps=n_steps)
        f0_list.append(res["f0_fit_hz"])
        d1_list.append(res["D1_fit_hz"])
        d2_list.append(res["D2_fit_hz"])
        veff_list.append(res["V_eff_travelling_m3"])
        ppw_list.append(res["ppw_core"])
        gap_cells_list.append(res["cells_across_gap"])
        cells_list.append(res["total_cells_2d"])
        times_list.append(res["wall_time_s"])

    # Locally refined graded mesh (dx_fine in ring walls/gap, graded in outer cladding/PML)
    res_graded = run_fit_cold_ring(dx_m=dx_levels_m[-1], anomalous=True, graded=True, n_steps=n_steps)

    # Richardson extrapolation p = 2 from Medium (dx_1) and Fine (dx_2)
    r_ratio = float(dx_levels_m[-2] / dx_levels_m[-1])
    f0_rich = float(f0_list[-1] + (f0_list[-1] - f0_list[-2]) / (r_ratio**2 - 1.0))
    d1_rich = float(d1_list[-1] + (d1_list[-1] - d1_list[-2]) / (r_ratio**2 - 1.0))
    d2_rich = float(d2_list[-1] + (d2_list[-1] - d2_list[-2]) / (r_ratio**2 - 1.0))
    veff_rich = float(veff_list[-1] + (veff_list[-1] - veff_list[-2]) / (r_ratio**2 - 1.0))

    f0_bessel = res["f0_bessel_hz"]
    d1_bessel = res["D1_bessel_hz"]
    d2_bessel = res["D2_bessel_hz"]

    base_dict = {
        "levels": level_names,
        "dx_nm": [float(d * 1e9) for d in dx_levels_m],
        "cells_2d": cells_list,
        "ppw_core": ppw_list,
        "cells_across_gap": gap_cells_list,
        "f0_hz": f0_list,
        "D1_hz": d1_list,
        "D2_hz": d2_list,
        "V_eff_m3": veff_list,
        "wall_times_s": times_list,
        "f0_richardson_hz": f0_rich,
        "D1_richardson_hz": d1_rich,
        "D2_richardson_hz": d2_rich,
        "V_eff_richardson_m3": veff_rich,
        "f0_bessel_hz": f0_bessel,
        "D1_bessel_hz": d1_bessel,
        "D2_bessel_hz": d2_bessel,
        "f0_rel_err_vs_bessel": [abs(f - f0_bessel) / f0_bessel for f in f0_list],
        "D1_rel_err_vs_bessel": [abs(d - d1_bessel) / d1_bessel for d in d1_list],
        "D2_rel_err_vs_bessel": [abs(d - d2_bessel) / abs(d2_bessel) for d in d2_list],
        "D2_richardson_rel_err": abs(d2_rich - d2_bessel) / abs(d2_bessel),
        "graded_cells_2d": res_graded["total_cells_2d"],
        "graded_cell_reduction_pct": float(
            100.0 * (1.0 - res_graded["total_cells_2d"] / cells_list[-1])
        ),
        "graded_f0_hz": res_graded["f0_fit_hz"],
        "graded_D1_hz": res_graded["D1_fit_hz"],
        "graded_D2_hz": res_graded["D2_fit_hz"],
        "graded_f0_rel_err": abs(res_graded["f0_fit_hz"] - f0_bessel) / f0_bessel,
        "graded_D2_rel_err": abs(res_graded["D2_fit_hz"] - d2_bessel) / abs(d2_bessel),
        "total_wall_time_s": float(time.perf_counter() - t0),
    }
    return augment_convergence_dict(base_dict)


def run_lle_detuning_scan(
    d2_fit_hz: float = 27.89e9,
    kappa_tot_hz: float = 401.06e9,
    f_pump: float | None = 2.350,
    n_modes: int = 256,
    *,
    f0_fit_hz: float = 214.34e12,
    n0_core: float = 2.0265,
    v_eff_trav_m3: float = 1.2899e-18,
    chi3_m2_v2: float = DEFAULT_CHI3_RING,
    eta_coupling: float = 0.9894,
    p_in_w: float | None = None,
) -> dict:
    """
    Run the FIT-informed Lugiato–Lefever Equation (LLE) split-step Fourier
    simulation (`wakis.solve_lle`) where ``d_2 = 2 D_2 / kappa`` and the
    parametric threshold ``P_th = hbar omega_0 kappa^2 / (8 g_0 eta)`` are
    derived directly from the 2D FIT cold-cavity dispersion (`D_2`, `f_0`,
    `V_eff,NL`), bus-coupled quality factor (`kappa`, `eta`), and Kerr
    susceptibility (`chi^(3)`). By default, the normalized external pump is
    chosen at ``f_pump = 2.350`` (inside the single-soliton step regime
    ``f > 1.241``), from which the required bus input power
    ``P_in = f_pump^2 * P_th`` follows; if ``f_pump=None`` and ``p_in_w`` is
    supplied, ``f_pump = sqrt(p_in_w / P_th)`` is computed instead.
    """
    t0 = time.perf_counter()
    # Normalized dispersion d_2 = 2 * D_2 / kappa from FIT D_2 and FIT kappa
    d2_norm = float(2.0 * d2_fit_hz / kappa_tot_hz)

    # Single-photon Kerr shift g_0 [rad/s] and parametric threshold power P_th [W]
    omega_0 = 2.0 * np.pi * float(f0_fit_hz)
    kappa_rad_s = 2.0 * np.pi * float(kappa_tot_hz)
    n2_m2_w = float(chi3_to_n2(chi3_m2_v2, eps_r=float(n0_core) ** 2))
    g0_rad_s = float(
        hbar * (omega_0**2) * c_light * n2_m2_w / ((float(n0_core) ** 2) * float(v_eff_trav_m3))
    )
    p_thresh_w = float(
        hbar * omega_0 * (kappa_rad_s**2) / (8.0 * g0_rad_s * float(eta_coupling))
    )
    if f_pump is None:
        p_in_val = 3.521 if p_in_w is None else float(p_in_w)
        f_pump = float(np.sqrt(p_in_val / p_thresh_w))
        p_in_w = p_in_val
    else:
        f_pump = float(f_pump)
        p_in_w = float((f_pump**2) * p_thresh_w)

    phi = np.linspace(-np.pi, np.pi, n_modes, endpoint=False)

    # 1. CW bistability curve across [-1.5, 7.5]
    zeta_grid = np.linspace(-1.5, 7.5, 181)
    bist = lle_cw_bistability(zeta_grid, f_pump=f_pump)

    # 2. Laser detuning scan from zeta_0 = -1.2 to +5.2 showing MI -> chaos -> soliton step
    n_scan = 4500
    dtau_scan = 0.012
    zeta_scan = np.linspace(-1.2, 5.2, n_scan)
    rng = np.random.default_rng(7)
    psi_seed = 0.02 * (rng.standard_normal(n_modes) + 1j * rng.standard_normal(n_modes))
    psi_seed += 0.35 * np.exp(-((phi / 0.18) ** 2))

    scan_res = solve_lle(
        d2=d2_norm,
        f_pump=f_pump,
        zeta_0=zeta_scan,
        n_modes=n_modes,
        dtau=dtau_scan,
        n_steps=n_scan,
        psi_init=psi_seed,
        record_every=25,
    )

    snaps = np.asarray(scan_res["snapshots"])
    n_snaps = snaps.shape[0]
    z_snaps = zeta_scan[np.linspace(24, n_scan - 1, n_snaps, dtype=int)]
    idx_mi = int(np.argmin(np.abs(z_snaps - 0.55)))
    idx_breath = int(np.argmin(np.abs(z_snaps - 1.75)))

    # 3. Stationary single soliton at zeta_0 = 4.2 (well inside [zeta_bist_min, zeta_max])
    zeta_sol = 4.2
    an_sol = lle_soliton_analytic(phi, zeta_0=zeta_sol, d2=d2_norm, f_pump=f_pump)
    sol_sim = solve_lle(
        d2=d2_norm,
        f_pump=f_pump,
        zeta_0=zeta_sol,
        n_modes=n_modes,
        dtau=0.01,
        n_steps=1600,
        psi_init=an_sol["psi"],
        noise_amp=0.0,
        record_every=40,
    )
    psi_sol = sol_sim["psi"]
    mu = sol_sim["mu"]
    spec_sol = sol_sim["spectrum"]

    # Measure simulated spectral FWHM (in mode numbers mu) from the background-subtracted soliton pulse
    psi_pulse = psi_sol - psi_sol[0]
    spec_pulse = np.abs(np.fft.fftshift(np.fft.fft(psi_pulse)) / float(n_modes)) ** 2
    sb_nonneg = mu >= 0
    mu_nonneg = mu[sb_nonneg].astype(float)
    p_nonneg = spec_pulse[sb_nonneg]
    p_norm = p_nonneg / p_nonneg[0]
    fwhm_mu_sim = 2.0 * float(np.interp(0.5, p_norm[::-1], mu_nonneg[::-1]))
    fwhm_mu_an = float(an_sol["fwhm_mu"])

    # Measure simulated azimuthal pulse FWHM (in radians) above CW background
    intensity_above_bg = np.abs(psi_sol - an_sol["psi_bg"]) ** 2
    half_peak = 0.5 * float(np.max(intensity_above_bg))
    right_mask = phi >= 0.0
    phi_right = phi[right_mask]
    int_right = intensity_above_bg[right_mask]
    fwhm_phi_sim = 2.0 * float(np.interp(half_peak, int_right[::-1], phi_right[::-1]))
    fwhm_phi_an = float(an_sol["fwhm_phi"])

    # 4. Dispersion scaling sweep: d2 in [0.02, 0.04, 0.08, 0.16] -> FWHM_mu ~ 1 / sqrt(d2)
    d2_sweep = [0.02, 0.04, 0.08, 0.16]
    fwhm_sweep_sim = []
    fwhm_sweep_an = []
    for d2_v in d2_sweep:
        an_v = lle_soliton_analytic(phi, zeta_0=zeta_sol, d2=d2_v, f_pump=f_pump)
        sim_v = solve_lle(
            d2=d2_v,
            f_pump=f_pump,
            zeta_0=zeta_sol,
            n_modes=n_modes,
            dtau=0.01,
            n_steps=1000,
            psi_init=an_v["psi"],
            noise_amp=0.0,
        )
        psi_p_v = sim_v["psi"] - sim_v["psi"][0]
        spec_p_v = np.abs(np.fft.fftshift(np.fft.fft(psi_p_v)) / float(n_modes)) ** 2
        p_v = spec_p_v[sb_nonneg]
        p_v_norm = p_v / p_v[0]
        fwhm_v = 2.0 * float(np.interp(0.5, p_v_norm[::-1], mu_nonneg[::-1]))
        fwhm_sweep_sim.append(fwhm_v)
        fwhm_sweep_an.append(float(an_v["fwhm_mu"]))

    # Fit log-log slope of fwhm_sweep_sim vs d2_sweep (analytic exponent is -0.5)
    disp_slope = float(np.polyfit(np.log(d2_sweep), np.log(fwhm_sweep_sim), 1)[0])

    return {
        "d2_fit_hz": float(d2_fit_hz),
        "kappa_tot_hz": float(kappa_tot_hz),
        "eta_coupling": float(eta_coupling),
        "g0_single_photon_hz": float(g0_rad_s / (2.0 * np.pi)),
        "p_thresh_w": p_thresh_w,
        "p_in_w": float(p_in_w),
        "d2_norm": d2_norm,
        "f_pump": float(f_pump),
        "zeta_sol": zeta_sol,
        "zeta_max_analytic": float(an_sol["zeta_max"]),
        "in_existence_range": bool(an_sol["in_existence_range"]),
        "phi": phi.tolist(),
        "mu": mu.tolist(),
        "zeta_bist_grid": zeta_grid.tolist(),
        "cw_lower": bist["lower_branch"].tolist(),
        "cw_middle": [None if np.isnan(x) else float(x) for x in bist["middle_branch"]],
        "cw_upper": [None if np.isnan(x) else float(x) for x in bist["upper_branch"]],
        "zeta_scan": zeta_scan[::10].tolist(),
        "power_scan": scan_res["power_trace"][::10].tolist(),
        "peak_scan": scan_res["peak_trace"][::10].tolist(),
        "psi_mi_abs2": (np.abs(snaps[idx_mi]) ** 2).tolist(),
        "psi_breather_abs2": (np.abs(snaps[idx_breath]) ** 2).tolist(),
        "psi_sol_abs2": (np.abs(psi_sol) ** 2).tolist(),
        "psi_sol_analytic_abs2": (np.abs(an_sol["psi"]) ** 2).tolist(),
        "spectrum_sol_sim": spec_sol.tolist(),
        "spectrum_sol_an": an_sol["spectrum_sech2"].tolist(),
        "fwhm_mu_sim": fwhm_mu_sim,
        "fwhm_mu_an": fwhm_mu_an,
        "fwhm_mu_rel_error": abs(fwhm_mu_sim - fwhm_mu_an) / fwhm_mu_an,
        "fwhm_phi_sim": fwhm_phi_sim,
        "fwhm_phi_an": fwhm_phi_an,
        "fwhm_phi_rel_error": abs(fwhm_phi_sim - fwhm_phi_an) / fwhm_phi_an,
        "d2_sweep": d2_sweep,
        "fwhm_sweep_sim": fwhm_sweep_sim,
        "fwhm_sweep_an": fwhm_sweep_an,
        "dispersion_scaling_exponent": disp_slope,
        "dispersion_exponent_rel_error": abs(disp_slope - (-0.5)) / 0.5,
        "scan_snapshots_abs2": (np.abs(snaps) ** 2).tolist(),
        "scan_zeta_snaps": z_snaps.tolist(),
        "wall_time_s": float(time.perf_counter() - t0),
    }


def overlay_microring_geometry(
    ax: plt.Axes,
    *,
    include_bus: bool = True,
    scale: float = 1e6,
    color: str = "#1a1a1a",
) -> None:
    """Overlay the microring inner/outer radii and bus waveguide outlines on a 2D axes."""
    r_in = (DEFAULT_RING_RADIUS_M - 0.5 * DEFAULT_RING_WIDTH_M) * scale
    r_out = (DEFAULT_RING_RADIUS_M + 0.5 * DEFAULT_RING_WIDTH_M) * scale
    for rad in (r_in, r_out):
        circ = Circle(
            (0.0, 0.0),
            rad,
            fill=False,
            edgecolor=color,
            linewidth=1.3,
            linestyle="-",
            zorder=6,
        )
        ax.add_patch(circ)
    if include_bus:
        y_lo = r_out + DEFAULT_GAP_M * scale
        w_b = DEFAULT_BUS_WIDTH_M * scale
        x_h = DEFAULT_DOMAIN_HALF_M * scale
        rect = Rectangle(
            (-x_h, y_lo),
            2.0 * x_h,
            w_b,
            fill=False,
            edgecolor=color,
            linewidth=1.3,
            linestyle="-",
            zorder=6,
        )
        ax.add_patch(rect)


def save_microcomb_setup_figures(fig_dir: Path) -> dict:
    """
    Generate `00a_setup_geometry.png`, `00b_setup_mesh_and_excitation.png`, and
    `00_setup_table.md` for the 2D bus-coupled Kerr microring resonator using
    `wakis.schematic.save_setup_figures`.
    """
    from wakis.templates.shapes import Box, Cylinder

    fig_dir.mkdir(parents=True, exist_ok=True)
    x_nodes, y_nodes = build_microring_graded_axes(dx_fine=25.0e-9)
    z_nodes = np.linspace(-0.5 * DEFAULT_SLAB_DZ_M, 0.5 * DEFAULT_SLAB_DZ_M, 3)
    grid = wakis.GridFIT3D(
        -DEFAULT_DOMAIN_HALF_M,
        DEFAULT_DOMAIN_HALF_M,
        -DEFAULT_DOMAIN_HALF_M,
        DEFAULT_DOMAIN_HALF_M,
        -0.5 * DEFAULT_SLAB_DZ_M,
        0.5 * DEFAULT_SLAB_DZ_M,
        len(x_nodes) - 1,
        len(y_nodes) - 1,
        2,
        x=x_nodes,
        y=y_nodes,
        z=z_nodes,
        verbose=0,
    )
    solver = wakis.SolverFIT3D(
        grid,
        cfln=0.5,
        bc_low=["cpml", "cpml", "periodic"],
        bc_high=["cpml", "cpml", "periodic"],
        n_pml=10,
        bg=[DEFAULT_EPS_CLAD, 1.0, 0.0],
        verbose=0,
    )
    x_e = solver.grid.x[: solver.Nx]
    y_e = solver.grid.y[: solver.Ny]
    fill_ring, fill_bus = _build_ring_fill_fraction(x_e, y_e, include_bus=True)
    fill_tot = np.clip(fill_ring + fill_bus, 0.0, 1.0)
    eps_map = DEFAULT_EPS_CLAD + fill_tot * (DEFAULT_EPS_INF_ANOM - DEFAULT_EPS_CLAD)
    for k in range(solver.Nz):
        solver.ieps[: solver.Nx, : solver.Ny, k, "z"] = 1.0 / (epsilon_0 * eps_map)

    mat_kerr = make_ring_material(anomalous=True, chi3=DEFAULT_CHI3_RING)
    r_in = DEFAULT_RING_RADIUS_M - 0.5 * DEFAULT_RING_WIDTH_M
    r_out = DEFAULT_RING_RADIUS_M + 0.5 * DEFAULT_RING_WIDTH_M
    y_bus_c = r_out + DEFAULT_GAP_M + 0.5 * DEFAULT_BUS_WIDTH_M

    bus_box = Box(
        center=(0.0, y_bus_c, 0.0),
        size=(2.0 * DEFAULT_DOMAIN_HALF_M, DEFAULT_BUS_WIDTH_M, DEFAULT_SLAB_DZ_M),
        material=DEFAULT_EPS_INF_ANOM,
    )
    bus_box.material = mat_kerr
    ring_outer = Cylinder(
        center=(0.0, 0.0, 0.0),
        radius=r_out,
        length=DEFAULT_SLAB_DZ_M,
        material=DEFAULT_EPS_INF_ANOM,
        axis="z",
    )
    ring_outer.material = mat_kerr
    ring_inner = Cylinder(
        center=(0.0, 0.0, 0.0),
        radius=r_in,
        length=DEFAULT_SLAB_DZ_M,
        material=DEFAULT_EPS_CLAD,
        axis="z",
    )

    f0_ref = 214.5e12
    src_pulse = GaussianSource(frequency=f0_ref, fwidth=35.0e12)
    src_sheet = CurrentSource(
        src_pulse,
        component="z",
        center=(-1.8e-6, y_bus_c, 0.0),
        size=(0.0, DEFAULT_BUS_WIDTH_M, DEFAULT_SLAB_DZ_M),
        amplitude=1.0,
    )
    src_sheet.label = "Bus Pump Port (Jz)"

    mon_through = wakis.DFTMonitor(
        np.linspace(170.0e12, 250.0e12, 41),
        components=("Ez",),
        center=(1.8e-6, y_bus_c, 0.0),
        size=(0.0, DEFAULT_BUS_WIDTH_M, DEFAULT_SLAB_DZ_M),
    )
    mon_through.label = "Through Port (DFTMonitor)"

    mon_ring = wakis.ResonanceMonitor(
        component="Ez",
        center=(0.0, -DEFAULT_RING_RADIUS_M, 0.0),
        size=(0.0, 0.0, 0.0),
    )
    mon_ring.label = "Ring WGM Probe (ResonanceMonitor)"

    out = save_setup_figures(
        solver,
        fig_dir,
        plane="xy",
        structures=[bus_box, ring_outer, ring_inner],
        material_names={
            "region_0": "Kerr Bus Waveguide",
            "region_1": "Kerr Microring Core",
            "region_2": "Inner Air Cladding",
        },
        sources=[src_sheet],
        monitors=[mon_through, mon_ring],
        unit="µm",
        wavelength=c_light / f0_ref,
        freq=f0_ref,
        min_feature=DEFAULT_GAP_M,
        title_geometry="2D TM Kerr Microring & Bus Coupler Setup (XY Plane)",
        title_mesh="Locally Refined Graded Mesh (1× Gap Zoom) & Pump Spectrum",
        variations=[
            {
                "name": "Cold-cavity dispersion extraction (task_0)",
                "details": "Azimuthal orders m ∈ {14..20} (m₀ = 17, f₀ = 214.5 THz), anomalous (D₂ > 0) vs normal (D₂ < 0)",
            },
            {
                "name": "Bus-coupled Q extraction & Kerr shift (task_1, task_2)",
                "details": "160 nm bus–ring gap (Q₀, Q_ex, κ) and short 2D FIT Kerr ring resonance shift vs V_eff,NL",
            },
            {
                "name": "3-level convergence + graded mesh (task_3)",
                "details": "Uniform Δx ∈ {30, 25, 20} nm (≥ 20 cells/λ_core) + locally refined graded mesh & Richardson extrapolation",
            },
            {
                "name": "FIT-informed Lugiato–Lefever microcomb (task_4)",
                "details": "256-mode split-step Fourier scan (MI → chaotic breather → single DKS soliton step) + sech² spectrum",
            },
        ],
    )
    return out


def generate_all_figures(
    cold_anom: dict,
    cold_norm: dict,
    bus_res: dict,
    kerr_res: dict,
    conv_res: dict,
    lle_res: dict,
    fig_dir: Path,
) -> list[Path]:
    """Generate all tutorial figures (`01_...` through `05_...`) and GIF (`06_...`)."""
    apply_style()
    fig_dir.mkdir(parents=True, exist_ok=True)
    c_blue, c_orange, c_green, c_gold, c_purple, c_sky, _, c_black = PALETTE
    written: list[Path] = []

    # ------------------------------------------------------------------
    # Figure 01: Cold-cavity WGM & Bus-Coupled E_z field maps + Q extraction
    # ------------------------------------------------------------------
    fig1, (ax1a, ax1b, ax1c) = plt.subplots(1, 3, figsize=(13.2, 4.1))
    x_um = cold_anom["x_e"] * 1e6
    y_um = cold_anom["y_e"] * 1e6
    snap_cold = cold_anom["snap_ez"]
    vmax_c = float(np.max(np.abs(snap_cold))) or 1.0
    im1 = ax1a.pcolormesh(
        x_um,
        y_um,
        (snap_cold / vmax_c).T,
        cmap=CMAP_DIVERGING,
        vmin=-1.0,
        vmax=1.0,
        shading="auto",
    )
    overlay_microring_geometry(ax1a, include_bus=False)
    ax1a.set_aspect("equal")
    ax1a.set_xlabel(r"$x$ [$\mu\mathrm{m}$]")
    ax1a.set_ylabel(r"$y$ [$\mu\mathrm{m}$]")
    ax1a.set_title(r"Cold-Cavity WGM $E_z(x, y)$", pad=8)
    plt.colorbar(im1, ax=ax1a, fraction=0.046, pad=0.04, label=r"$E_z / E_{\max}$ [-]")
    panel_label(ax1a, "(a)")

    xb_um = bus_res["x_e"] * 1e6
    yb_um = bus_res["y_e"] * 1e6
    snap_b = bus_res["snap_bus_ez"]
    vmax_b = float(np.max(np.abs(snap_b))) or 1.0
    im2 = ax1b.pcolormesh(
        xb_um,
        yb_um,
        (snap_b / vmax_b).T,
        cmap=CMAP_DIVERGING,
        vmin=-1.0,
        vmax=1.0,
        shading="auto",
    )
    overlay_microring_geometry(ax1b, include_bus=True)
    ax1b.set_aspect("equal")
    ax1b.set_xlabel(r"$x$ [$\mu\mathrm{m}$]")
    ax1b.set_ylabel(r"$y$ [$\mu\mathrm{m}$]")
    ax1b.set_title(rf"Bus-Coupled Ring ($g = {bus_res['gap_m']*1e9:.0f}\;\mathrm{{nm}}$)", pad=8)
    plt.colorbar(im2, ax=ax1b, fraction=0.046, pad=0.04, label=r"$E_z / E_{\max}$ [-]")
    panel_label(ax1b, "(b)")

    spectrum_grid(ax1c)
    mu_arr = np.array(cold_anom["mu"])
    ax1c.plot(
        mu_arr,
        np.array(cold_anom["f_bessel_hz"]) * 1e-12,
        "-",
        color=c_blue,
        lw=2.0,
        label="Cylindrical Bessel theory",
    )
    ax1c.plot(
        mu_arr,
        np.array(cold_anom["f_fit_hz"]) * 1e-12,
        "o",
        color=c_orange,
        ms=6.5,
        label="2D FIT + harminv",
    )
    ax1c.set_xlabel(r"Relative azimuthal order $\mu = m - m_0$ [-]")
    ax1c.set_ylabel(r"Resonance frequency $f_\mu$ [THz]")
    ax1c.set_title(rf"WGM Comb Family ($\mathrm{{FSR}} = {cold_anom['D1_fit_hz']*1e-12:.2f}\;\mathrm{{THz}}$)")
    ax1c.legend(loc="upper left", fontsize=8.5)
    panel_label(ax1c, "(c)")

    fig1.tight_layout()
    p1 = fig_dir / "01_microring_wgm_fields_and_resonances.png"
    save(fig1, p1)
    plt.close(fig1)
    written.append(p1)

    # ------------------------------------------------------------------
    # Figure 02: Integrated dispersion D_int(mu) (anomalous vs normal) & Kerr shift
    # ------------------------------------------------------------------
    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(11.2, 4.4))
    spectrum_grid(ax2a)
    spectrum_grid(ax2b)

    mu_dense = np.linspace(-3.2, 3.2, 100)
    ax2a.plot(
        mu_dense,
        0.5 * (cold_anom["D2_bessel_hz"] * 1e-9) * (mu_dense**2),
        color=c_blue,
        lw=2.0,
        label=rf"Bessel anomalous ($D_2/2\pi = {cold_anom['D2_bessel_hz']*1e-9:+.1f}\;\mathrm{{GHz}}$)",
    )
    ax2a.plot(
        mu_arr,
        np.array(cold_anom["D_int_fit_hz"]) * 1e-9,
        "o",
        color=c_orange,
        ms=6.5,
        label=rf"2D FIT anomalous ($D_2/2\pi = {cold_anom['D2_fit_hz']*1e-9:+.1f}\;\mathrm{{GHz}}$)",
    )
    ax2a.plot(
        mu_dense,
        0.5 * (cold_norm["D2_bessel_hz"] * 1e-9) * (mu_dense**2),
        color=c_green,
        ls="--",
        lw=1.8,
        label=rf"Bessel normal ($D_2/2\pi = {cold_norm['D2_bessel_hz']*1e-9:+.1f}\;\mathrm{{GHz}}$)",
    )
    ax2a.plot(
        mu_arr,
        np.array(cold_norm["D_int_fit_hz"]) * 1e-9,
        "s",
        color=c_purple,
        ms=6.0,
        label=rf"2D FIT normal ($D_2/2\pi = {cold_norm['D2_fit_hz']*1e-9:+.1f}\;\mathrm{{GHz}}$)",
    )
    ax2a.axhline(0.0, color=c_black, lw=0.8)
    ax2a.set_xlabel(r"Relative mode number $\mu = m - m_0$ [-]")
    ax2a.set_ylabel(r"Integrated dispersion $D_{\mathrm{int}}(\mu) / 2\pi$ [GHz]")
    ax2a.set_title(r"Integrated Dispersion $D_{\mathrm{int}}(\mu) = (D_2 / 2)\mu^2$")
    ax2a.legend(loc="upper center", fontsize=8.0)
    panel_label(ax2a, "(a)")

    u_fj = np.array(kerr_res["stored_energy_j"]) * 1e15
    u_dense = np.linspace(0.0, float(np.max(u_fj)) * 1.08, 100)
    slope_kerr = kerr_res["df_analytic_hz"][-1] * 1e-9 / u_fj[-1]
    ax2b.plot(
        u_dense,
        slope_kerr * u_dense,
        color=c_blue,
        lw=2.0,
        label=rf"Perturbation theory ($V_{{\mathrm{{eff,NL}}}} = {kerr_res['V_eff_nl_um3']:.2f}\;\mu\mathrm{{m}}^3$)",
    )
    ax2b.plot(
        u_Dense := u_dense,
        slope_kerr * 0.5 * u_Dense,
        color=c_black,
        ls="--",
        lw=1.5,
        label=r"Wrong mode volume ($2\times V_{\mathrm{eff,NL}}$, fails)",
    )
    ax2b.plot(
        u_fj,
        np.array(kerr_res["df_fit_hz"]) * 1e-9,
        "o",
        color=c_orange,
        ms=6.5,
        label=r"2D FIT Kerr ring + harminv ($\chi^{(3)} > 0$)",
    )
    ax2b.set_xlabel(r"Stored intracavity mode energy $U$ [fJ]")
    ax2b.set_ylabel(r"Nonlinear resonance shift $\Delta f_{\mathrm{NL}}$ [GHz]")
    ax2b.set_title("2D Microring Kerr Resonance Shift Verification")
    ax2b.legend(loc="lower left", fontsize=8.2)
    panel_label(ax2b, "(b)")

    fig2.tight_layout()
    p2 = fig_dir / "02_dispersion_and_kerr_ring_shift.png"
    save(fig2, p2)
    plt.close(fig2)
    written.append(p2)

    # ------------------------------------------------------------------
    # Figure 03: Log-log resolution convergence study + O(dx^2) guide
    # ------------------------------------------------------------------
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(11.2, 4.4))
    spectrum_grid(ax3a)
    spectrum_grid(ax3b)

    dx_nm = np.array(conv_res["dx_nm"])
    err_f0 = np.array(conv_res["f0_rel_err_vs_bessel"]) * 100.0
    err_d1 = np.array(conv_res["D1_rel_err_vs_bessel"]) * 100.0
    err_d2 = np.array(conv_res["D2_rel_err_vs_bessel"]) * 100.0

    ax3a.loglog(dx_nm, err_f0, "o-", color=c_blue, lw=2.0, ms=6.5, label=r"Resonance $f_0$ error")
    ax3a.loglog(dx_nm, err_d1, "s-", color=c_orange, lw=2.0, ms=6.0, label=r"FSR $D_1/2\pi$ error")
    ax3a.loglog(dx_nm, err_d2, "^-", color=c_green, lw=2.0, ms=6.5, label=r"Dispersion $D_2/2\pi$ error")
    # O(dx^2) reference guide
    dx_guide = np.linspace(dx_nm[-1] * 0.95, dx_nm[0] * 1.05, 50)
    guide_y = err_f0[-1] * ((dx_guide / dx_nm[-1]) ** 2)
    ax3a.loglog(dx_guide, guide_y, ":", color=c_black, lw=1.6, label=r"$\mathcal{O}(\Delta x^2)$ order guide")
    ax3a.plot(
        [dx_nm[-1]],
        [conv_res["graded_f0_rel_err"] * 100.0],
        "*",
        color=c_purple,
        ms=10.0,
        label=rf"Graded mesh ($-{conv_res['graded_cell_reduction_pct']:.1f}\%$ cells)",
    )
    ax3a.set_xlabel(r"Grid spacing $\Delta x$ [nm]")
    ax3a.set_ylabel("Relative error vs Bessel theory [%]")
    ax3a.set_title(r"2D FIT Resolution Convergence ($\geq 20\;\mathrm{cells}/\lambda$)")
    ax3a.legend(loc="best", fontsize=8.0)
    panel_label(ax3a, "(a)")

    conv_res = augment_convergence_dict(conv_res)
    levels_x = np.arange(len(conv_res["levels"]) + 2)
    d2_bars = [x * 1e-9 for x in conv_res["D2_hz"]] + [
        conv_res["graded_D2_hz"] * 1e-9,
        conv_res["D2_richardson_hz"] * 1e-9,
    ]
    labels_x = (
        [f"{lvl}\n({dx:.0f} nm)" for lvl, dx in zip(conv_res["levels"], conv_res["dx_nm"])]
        + [
            f"Graded\n(-{conv_res['graded_cell_reduction_pct']:.0f}% cells)",
            "Richardson\n($\\Delta x \\to 0$)",
        ]
    )
    ax3b.plot(
        levels_x,
        d2_bars,
        "o-",
        color=c_green,
        lw=2.0,
        ms=7.0,
        label=r"2D FIT $D_2/(2\pi)$ + Graded + Richardson ($p=2$)",
    )
    d2_obs_ghz = conv_res["D2_richardson_obs_hz"] * 1e-9
    p_obs_d2 = conv_res["D2_observed_order"]
    ax3b.plot(
        [levels_x[-1]],
        [d2_obs_ghz],
        "D",
        color=c_purple,
        ms=7.0,
        label=rf"Richardson ($p_{{\mathrm{{obs}}}} = {p_obs_d2:.2f}$: ${d2_obs_ghz:+.2f}\;\mathrm{{GHz}}$)",
    )
    ax3b.vlines(
        levels_x[-1],
        min(d2_obs_ghz, d2_bars[-1]),
        max(d2_obs_ghz, d2_bars[-1]),
        color=c_purple,
        lw=2.5,
        alpha=0.55,
    )
    ax3b.axhline(
        conv_res["D2_bessel_hz"] * 1e-9,
        color=c_orange,
        ls="--",
        lw=1.8,
        label=rf"Exact Bessel $D_2/(2\pi) = {conv_res['D2_bessel_hz']*1e-9:+.2f}\;\mathrm{{GHz}}$",
    )
    d2_rich_err_pct = abs(conv_res["D2_richardson_hz"] - conv_res["D2_bessel_hz"]) / abs(conv_res["D2_bessel_hz"]) * 100.0
    d2_obs_err_pct = conv_res["D2_richardson_obs_rel_err"] * 100.0
    ax3b.set_xticks(levels_x)
    ax3b.set_xticklabels(labels_x, fontsize=8.0)
    ax3b.set_ylabel(r"Anomalous dispersion $D_2 / (2\pi)$ [GHz]")
    ax3b.set_title(
        rf"$D_2/(2\pi)$ Convergence (Richardson Band: {d2_rich_err_pct:.2f}%–{d2_obs_err_pct:.2f}% Error)"
    )
    ax3b.legend(loc="lower right", fontsize=7.8)
    panel_label(ax3b, "(b)")

    fig3.tight_layout()
    p3 = fig_dir / "03_resolution_convergence.png"
    save(fig3, p3)
    plt.close(fig3)
    written.append(p3)

    # ------------------------------------------------------------------
    # Figure 04: LLE CW bistability, detuning scan (soliton step), & MI -> Breather -> DKS
    # ------------------------------------------------------------------
    fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(11.2, 4.4))
    spectrum_grid(ax4a)
    spectrum_grid(ax4b)

    z_bist = np.array(lle_res["zeta_bist_grid"])
    cw_low = np.array(lle_res["cw_lower"])
    cw_mid = np.array([np.nan if v is None else v for v in lle_res["cw_middle"]])
    cw_up = np.array([np.nan if v is None else v for v in lle_res["cw_upper"]])

    ax4a.plot(z_bist, cw_low, color=c_blue, lw=1.8, label="CW lower branch (stable)")
    ax4a.plot(z_bist, cw_mid, "--", color=c_black, lw=1.4, label="CW middle branch (unstable)")
    ax4a.plot(z_bist, cw_up, color=c_green, lw=1.8, label="CW upper branch (MI unstable)")
    ax4a.plot(
        lle_res["zeta_scan"],
        lle_res["power_scan"],
        color=c_orange,
        lw=1.6,
        alpha=0.9,
        label=r"LLE scan $\langle|\Psi|^2\rangle(\zeta_0)$ (soliton step)",
    )
    ax4a.axvline(
        lle_res["zeta_max_analytic"],
        color=c_purple,
        ls=":",
        lw=1.5,
        label=rf"$\zeta_{{\max}} = (\pi^2/8)f^2 = {lle_res['zeta_max_analytic']:.2f}$",
    )
    ax4a.set_xlabel(r"Normalized pump detuning $\zeta_0 = 2(\omega_0 - \omega_p)/\kappa$ [-]")
    ax4a.set_ylabel(r"Mean intracavity power $\langle|\Psi|^2\rangle$ [-]")
    ax4a.set_title("LLE Bistability & Pump-Detuning Scan")
    ax4a.legend(loc="upper left", fontsize=7.8)
    panel_label(ax4a, "(a)")

    phi_arr = np.array(lle_res["phi"])
    ax4b.plot(
        phi_arr,
        lle_res["psi_mi_abs2"],
        color=c_green,
        lw=1.4,
        alpha=0.85,
        label=r"Primary MI rolls ($\zeta_0 \approx 0.55$)",
    )
    ax4b.plot(
        phi_arr,
        lle_res["psi_breather_abs2"],
        color=c_purple,
        lw=1.4,
        alpha=0.8,
        label=r"Chaotic / breather MI ($\zeta_0 \approx 1.75$)",
    )
    ax4b.plot(
        phi_arr,
        lle_res["psi_sol_abs2"],
        color=c_orange,
        lw=2.2,
        label=rf"Single DKS SSFM ($\zeta_0 = {lle_res['zeta_sol']:.1f}$)",
    )
    ax4b.plot(
        phi_arr,
        lle_res["psi_sol_analytic_abs2"],
        "--",
        color=c_blue,
        lw=1.6,
        label=r"Analytic $\mathrm{sech}(\phi/\Delta\phi_{\mathrm{sol}})$",
    )
    ax4b.set_xlim(-np.pi, np.pi)
    ax4b.set_xlabel(r"Co-rotating azimuthal angle $\phi$ [rad]")
    ax4b.set_ylabel(r"Intracavity intensity $|\Psi(\phi)|^2$ [-]")
    ax4b.set_title(r"MI $\to$ Breather $\to$ Single Dissipative Kerr Soliton")
    ax4b.legend(loc="upper right", fontsize=8.0)
    panel_label(ax4b, "(b)")

    fig4.tight_layout()
    p4 = fig_dir / "04_lle_detuning_scan_and_soliton_regimes.png"
    save(fig4, p4)
    plt.close(fig4)
    written.append(p4)

    # ------------------------------------------------------------------
    # Figure 05: Soliton comb sech^2 spectrum & bandwidth vs sqrt(D2) scaling
    # ------------------------------------------------------------------
    fig5, (ax5a, ax5b) = plt.subplots(1, 2, figsize=(11.2, 4.4))
    spectrum_grid(ax5a)
    spectrum_grid(ax5b)

    mu_lle = np.array(lle_res["mu"])
    m_sb = (np.abs(mu_lle) >= 1) & (np.abs(mu_lle) <= 80)
    spec_sim_db = 10.0 * np.log10(np.maximum(np.array(lle_res["spectrum_sol_sim"])[m_sb], 1e-12))
    spec_an_db = 10.0 * np.log10(np.maximum(np.array(lle_res["spectrum_sol_an"])[m_sb], 1e-12))

    ax5a.vlines(
        mu_lle[m_sb],
        -70.0,
        spec_sim_db,
        color=c_blue,
        lw=0.9,
        alpha=0.65,
        label=rf"LLE SSFM comb lines ($\Delta\mu = {lle_res['fwhm_mu_sim']:.1f}$)",
    )
    ax5a.plot(
        mu_lle[m_sb],
        spec_an_db,
        color=c_orange,
        lw=2.2,
        label=rf"Analytic $\mathrm{{sech}}^2$ envelope ($\Delta\mu = {lle_res['fwhm_mu_an']:.1f}$)",
    )
    ax5a.set_xlim(-80, 80)
    ax5a.set_ylim(-68, -12)
    ax5a.set_xlabel(r"Relative comb line index $\mu = m - m_0$ [-]")
    ax5a.set_ylabel(r"Comb line power $|\Psi_\mu|^2$ [dB]")
    ax5a.set_title(r"FIT-Informed Soliton Microcomb Spectrum ($\mathrm{sech}^2$)")
    ax5a.legend(loc="upper right", fontsize=8.2)
    panel_label(ax5a, "(a)")

    d2_sw = np.array(lle_res["d2_sweep"])
    fwhm_sw_sim = np.array(lle_res["fwhm_sweep_sim"])
    fwhm_sw_an = np.array(lle_res["fwhm_sweep_an"])
    d2_dense = np.geomspace(d2_sw[0] * 0.85, d2_sw[-1] * 1.15, 100)
    scale_ref = fwhm_sw_an[0] * np.sqrt(d2_sw[0])

    ax5b.loglog(
        d2_dense,
        scale_ref / np.sqrt(d2_dense),
        color=c_blue,
        lw=2.0,
        label=r"Analytic $\Delta\mu_{\mathrm{FWHM}} \propto d_2^{-1/2}$",
    )
    ax5b.loglog(
        d2_sw,
        fwhm_sw_sim,
        "o",
        color=c_orange,
        ms=7.0,
        label=rf"LLE SSFM (fitted slope $= {lle_res['dispersion_scaling_exponent']:.3f}$)",
    )
    ax5b.set_xlabel(r"Normalized anomalous dispersion $d_2 = 2 D_2 / \kappa$ [-]")
    ax5b.set_ylabel(r"3 dB spectral bandwidth $\Delta\mu_{\mathrm{FWHM}}$ [modes]")
    ax5b.set_title(r"Soliton Bandwidth Scaling $\Delta\mu_{\mathrm{FWHM}} \propto D_2^{-1/2}$")
    ax5b.legend(loc="upper right", fontsize=8.2)
    panel_label(ax5b, "(b)")

    fig5.tight_layout()
    p5 = fig_dir / "05_soliton_comb_spectrum_and_d2_scaling.png"
    save(fig5, p5)
    plt.close(fig5)
    written.append(p5)

    # ------------------------------------------------------------------
    # Figure 06: Animated GIF of intracavity soliton formation & 2D ring wave
    # ------------------------------------------------------------------
    gif_path = generate_microcomb_gif(cold_anom, lle_res, fig_dir)
    written.append(gif_path)

    return written


def generate_microcomb_gif(cold_anom: dict, lle_res: dict, fig_dir: Path) -> Path:
    """
    Create an animated GIF (`figures/kerr_microcomb_dynamics.gif`, < 5 MB) showing
    the 2D microring intracavity intensity `|E_z(x, y)|^2` reconstructed from the
    evolving LLE envelope `Psi(phi, tau) * u_{m0}(r)` alongside the azimuthal pulse
    profile `|Psi(phi, tau)|^2` across the MI -> breather -> single-soliton scan.
    """
    apply_style()
    c_blue, c_orange, _, _, _, _, _, _ = PALETTE

    x_um = cold_anom["x_e"] * 1e6
    y_um = cold_anom["y_e"] * 1e6
    xx, yy = np.meshgrid(x_um, y_um, indexing="ij")
    phi_2d = np.arctan2(yy, xx)
    rad_prof = cold_anom["radial_m0_profile"]
    phi_1d = np.array(lle_res["phi"])

    snaps_all = np.asarray(lle_res["scan_snapshots_abs2"])
    z_all = np.asarray(lle_res["scan_zeta_snaps"])
    # Select 24 evenly spaced frames + hold on the final single-soliton state
    idx_frames = np.linspace(0, len(snaps_all) - 1, 20, dtype=int).tolist() + [
        len(snaps_all) - 1
    ] * 4
    sol_abs2 = np.asarray(lle_res["psi_sol_abs2"])
    an_abs2 = np.asarray(lle_res["psi_sol_analytic_abs2"])

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(9.4, 4.0), dpi=110)
    spectrum_grid(ax_r)

    init_2d = (rad_prof**2) * np.interp(phi_2d, phi_1d, sol_abs2, period=2.0 * np.pi)
    im = ax_l.pcolormesh(
        x_um,
        y_um,
        init_2d.T,
        cmap=CMAP_INTENSITY,
        vmin=0.0,
        vmax=float(np.max(sol_abs2)),
        shading="auto",
    )
    overlay_microring_geometry(ax_l, include_bus=True, color="#ffffff")
    ax_l.set_aspect("equal")
    ax_l.set_xlabel(r"$x$ [$\mu\mathrm{m}$]")
    ax_l.set_ylabel(r"$y$ [$\mu\mathrm{m}$]")
    title_l = ax_l.set_title(r"2D Ring Intensity $|\Psi(\phi, \tau) u_{m_0}(r)|^2$")

    (line_sim,) = ax_r.plot(phi_1d, sol_abs2, color=c_orange, lw=2.0, label=r"LLE $|\Psi(\phi, \tau)|^2$")
    ax_r.plot(phi_1d, an_abs2, "--", color=c_blue, lw=1.4, alpha=0.8, label="Target single DKS")
    ax_r.set_xlim(-np.pi, np.pi)
    ax_r.set_ylim(0.0, float(np.max(sol_abs2)) * 1.35)
    ax_r.set_xlabel(r"Azimuthal angle $\phi$ [rad]")
    ax_r.set_ylabel(r"Intensity $|\Psi(\phi, \tau)|^2$ [-]")
    title_r = ax_r.set_title(r"Pump Detuning Scan $\zeta_0(\tau)$")
    ax_r.legend(loc="upper right", fontsize=8.0)
    fig.tight_layout()

    def _update(frame_i: int):
        if frame_i < 20:
            k = idx_frames[frame_i]
            prof_1d = snaps_all[k]
            z_val = float(z_all[k])
            regime = "CW / Primary MI" if z_val < 1.0 else ("Chaotic Breather MI" if z_val < 2.6 else "Soliton Step")
        else:
            prof_1d = sol_abs2
            z_val = float(lle_res["zeta_sol"])
            regime = "Stationary Single DKS"
        map_2d = (rad_prof**2) * np.interp(phi_2d, phi_1d, prof_1d, period=2.0 * np.pi)
        im.set_array(map_2d.T.ravel())
        line_sim.set_ydata(prof_1d)
        title_l.set_text(rf"2D Ring Intensity ($\zeta_0 = {z_val:+.2f}$)")
        title_r.set_text(f"{regime} (detuning $\\zeta_0 = {z_val:+.2f}$)")
        return im, line_sim, title_l, title_r

    anim = FuncAnimation(fig, _update, frames=len(idx_frames), interval=120, blit=False)
    gif_path = fig_dir / "kerr_microcomb_dynamics.gif"
    anim.save(gif_path, writer=PillowWriter(fps=8))
    plt.close(fig)
    return gif_path
