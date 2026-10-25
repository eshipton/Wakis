"""Helper functions for Tutorial 3.3: N-Slit Diffraction Gratings and Unit-Cell Equivalence."""

from __future__ import annotations

from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c as c0
from scipy.signal import argrelextrema

from wakis import DFTMonitor, FluxMonitor, GridFIT3D, SolverFIT3D, figstyle, schematic
from wakis.diffraction import DiffractionMonitor
from wakis.estimate import estimate, preflight
from wakis.near2far import Near2FarMonitor
from wakis.sources import CurrentSource, GaussianSource
from wakis.templates.meshing import graded_axis
from wakis.templates.shapes import Box, assign_materials

DEFAULT_WAVELENGTH = 0.6e-6  # 600 nm (500 THz)


def sinc_phys(u: np.ndarray | float) -> np.ndarray:
    """Return the unnormalized sinc function sin(u) / u [-]."""
    u_arr = np.asarray(u, dtype=float)
    return np.sinc(u_arr / np.pi)


def n_slit_interference_factor(
    theta: np.ndarray,
    n_slits: int,
    period: float,
    wavelength: float = DEFAULT_WAVELENGTH,
) -> np.ndarray:
    """Normalized N-slit interference array factor (sin(N * alpha) / (N * sin(alpha)))^2.

    Parameters
    ----------
    theta : ndarray
        Diffraction angle measured from the +y normal [rad].
    n_slits : int
        Number of slits N >= 1 [-].
    period : float
        Center-to-center grating period d [m].
    wavelength : float, optional
        Free-space wavelength lambda [m].

    Returns
    -------
    ndarray
        Normalized array factor in [0, 1] [-].
    """
    if int(n_slits) < 1:
        raise ValueError(f"n_slits must be >= 1, got {n_slits}.")
    n = int(n_slits)
    theta_arr = np.asarray(theta, dtype=float)
    alpha = np.pi * (period / wavelength) * np.sin(theta_arr)
    denom = n * np.sin(alpha)
    num = np.sin(n * alpha)
    out = np.ones_like(alpha, dtype=float)
    mask = np.abs( np.sin(alpha) ) > 1e-12
    out[mask] = (num[mask] / denom[mask]) ** 2
    return out


def single_slit_envelope(
    theta: np.ndarray,
    slit_width: float,
    wavelength: float = DEFAULT_WAVELENGTH,
    obliquity: str = "rs1",
) -> np.ndarray:
    """Single-slit diffraction envelope sinc^2(beta) with optional obliquity factor."""
    theta_arr = np.asarray(theta, dtype=float)
    beta = np.pi * (slit_width / wavelength) * np.sin(theta_arr)
    base = sinc_phys(beta) ** 2
    key = obliquity.lower()
    if key == "rs1":
        return (np.cos(theta_arr) ** 2) * base
    if key in ("rs2", "none", "fraunhofer"):
        return base
    if key == "kirchhoff":
        return (0.5 * (1.0 + np.cos(theta_arr))) ** 2 * base
    raise ValueError(f"Unknown obliquity {obliquity!r}.")


def n_slit_farfield_pattern(
    theta: np.ndarray,
    n_slits: int,
    slit_width: float,
    period: float,
    wavelength: float = DEFAULT_WAVELENGTH,
    obliquity: str = "rs1",
) -> np.ndarray:
    """Normalized N-slit far-field angular intensity pattern."""
    return n_slit_interference_factor(theta, n_slits, period, wavelength) * single_slit_envelope(
        theta, slit_width, wavelength, obliquity=obliquity
    )


def grating_equation_angles(
    period: float,
    wavelength: float = DEFAULT_WAVELENGTH,
    theta_inc: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return propagating diffraction orders m and angles theta_m [rad] from d*(sin(theta_m) - sin(theta_i)) = m*lambda."""
    if period <= 0.0 or wavelength <= 0.0:
        raise ValueError("period > 0 and wavelength > 0 are required.")
    s_inc = float(np.sin(theta_inc))
    m_min = int(np.ceil((-1.0 + 1e-9 - s_inc) * period / wavelength))
    m_max = int(np.floor((1.0 - 1e-9 - s_inc) * period / wavelength))
    orders = np.arange(m_min, m_max + 1, dtype=int)
    sin_th = s_inc + orders * (wavelength / period)
    return orders, np.arcsin(sin_th)


def principal_peak_halfwidth_sin_theta(
    n_slits: int,
    period: float,
    wavelength: float = DEFAULT_WAVELENGTH,
) -> float:
    """Half-width from a principal maximum to the first adjacent zero: Delta(sin theta)_hw = lambda / (N * d) [-]."""
    if int(n_slits) < 1 or period <= 0.0 or wavelength <= 0.0:
        raise ValueError("n_slits >= 1, period > 0, and wavelength > 0 are required.")
    return float(wavelength / (int(n_slits) * period))


def principal_peak_hwzm_sin_theta(
    n_slits: int,
    period: float,
    wavelength: float = DEFAULT_WAVELENGTH,
) -> float:
    """Half-width from a principal maximum to the first adjacent zero: Delta(sin theta)_HWZM = lambda / (N * d) [-]."""
    return principal_peak_halfwidth_sin_theta(n_slits, period, wavelength)


def principal_peak_fwhm_sin_theta(
    n_slits: int,
    period: float,
    wavelength: float = DEFAULT_WAVELENGTH,
) -> float:
    """Asymptotic full width at half maximum in sin(theta): Delta(sin theta)_FWHM = 0.88589 * lambda / (N * d) [-]."""
    if int(n_slits) < 1 or period <= 0.0 or wavelength <= 0.0:
        raise ValueError("n_slits >= 1, period > 0, and wavelength > 0 are required.")
    return float(0.8858929 * wavelength / (int(n_slits) * period))


def secondary_maxima_count(n_slits: int) -> int:
    """Return the exact number (N - 2) of secondary maxima between consecutive principal orders [-]."""
    if int(n_slits) < 1:
        raise ValueError(f"n_slits must be >= 1, got {n_slits}.")
    return max(int(n_slits) - 2, 0)


def chromatic_resolving_power(n_slits: int, order: int) -> int:
    """Chromatic resolving power R = lambda / Delta_lambda = |m| * N [-]."""
    if int(n_slits) < 1 or int(order) == 0:
        raise ValueError("Require n_slits >= 1 and non-zero diffraction order m.")
    return int(abs(int(order)) * int(n_slits))


def rayleigh_resolving_power(n_slits: int, order_m: int) -> int:
    """Rayleigh chromatic resolving power R = lambda / Delta_lambda = |m| * N [-]."""
    return chromatic_resolving_power(n_slits, order_m)


def count_secondary_maxima_between_orders(
    theta_deg: np.ndarray,
    pattern: np.ndarray,
    theta_m0_deg: float,
    theta_m1_deg: float,
    margin_deg: float = 2.0,
) -> tuple[int, int]:
    """Count secondary maxima and minima strictly between two adjacent principal maxima.

    Parameters
    ----------
    theta_deg : ndarray
        Angular grid [deg].
    pattern : ndarray
        Angular intensity pattern (ideally divided by the smooth single-slit envelope).
    theta_m0_deg, theta_m1_deg : float
        Angles of two adjacent principal maxima (e.g. m=0 at 0 deg and m=1) [deg].
    margin_deg : float, optional
        Half-width margin around each principal peak excluded from the secondary-peak search [deg].

    Returns
    -------
    n_secondary_max : int
        Number of secondary maxima strictly between the two principal peaks (theory: N - 2).
    n_minima : int
        Number of interference minima strictly between the two principal peaks (theory: N - 1).
    """
    lo = min(theta_m0_deg, theta_m1_deg) + margin_deg
    hi = max(theta_m0_deg, theta_m1_deg) - margin_deg
    mask = (theta_deg >= lo) & (theta_deg <= hi)
    sub = np.asarray(pattern[mask], dtype=float)
    max_idx = argrelextrema(sub, np.greater, order=2)[0]
    min_idx = argrelextrema(sub, np.less, order=2)[0]
    return int(max_idx.size), int(min_idx.size)


def build_n_slit_grid(
    n_slits: int,
    slit_width: float,
    period: float,
    wavelength: float = DEFAULT_WAVELENGTH,
    screen_thickness: float = 0.15 * DEFAULT_WAVELENGTH,
    x_half_span: float | None = None,
    y_upstream: float = 1.2 * DEFAULT_WAVELENGTH,
    y_downstream: float = 2.4 * DEFAULT_WAVELENGTH,
    cpw: int = 22,
    min_slit_cells: int = 6,
    min_barrier_cells: int = 4,
    min_screen_cells: int = 4,
    n_pml: int = 10,
) -> dict:
    """Build a symmetric 2D graded FIT mesh snapped to all N slits and N-1 interior bars."""
    n = int(n_slits)
    if n < 1:
        raise ValueError(f"n_slits must be >= 1, got {n_slits}.")
    bridge_width = period - slit_width
    if n > 1 and bridge_width <= 0.0:
        raise ValueError("Require period > slit_width for N >= 2.")

    centers = (np.arange(n, dtype=float) - 0.5 * (n - 1)) * period
    x_outer = float(centers[-1] + 0.5 * slit_width)
    if x_half_span is None:
        x_half_span = max(5.0 * wavelength, x_outer + 2.4 * wavelength)

    dx_base = wavelength / float(cpw)
    dx_slit = min(dx_base, slit_width / float(max(min_slit_cells, 4)))
    dx_bridge = (
        min(dx_base, bridge_width / float(max(min_barrier_cells, 4)))
        if n > 1
        else dx_base
    )
    dy_screen = min(dx_base, screen_thickness / float(max(min_screen_cells, 4)))

    x_pts = [-float(x_half_span)]
    x_sizes = []
    x_min_c = []
    for k, xc in enumerate(centers):
        x_pts.append(float(xc - 0.5 * slit_width))
        x_pts.append(float(xc + 0.5 * slit_width))
        if k == 0:
            x_sizes.append(dx_base)
            x_min_c.append(4)
        else:
            x_sizes.append(dx_bridge)
            x_min_c.append(max(min_barrier_cells, 4))
        x_sizes.append(dx_slit)
        x_min_c.append(max(min_slit_cells, 4))
    x_pts.append(float(x_half_span))
    x_sizes.append(dx_base)
    x_min_c.append(4)

    x_core = graded_axis(x_pts, x_sizes, min_cells=x_min_c, grading=1.4)
    x_core = 0.5 * (x_core - x_core[::-1])

    half_t = 0.5 * screen_thickness
    y_core = graded_axis(
        [-y_upstream, -half_t, half_t, y_downstream],
        [dx_base, dy_screen, dx_base],
        min_cells=[4, max(min_screen_cells, 4), 4],
        grading=1.4,
    )

    dx_left = float(x_core[1] - x_core[0])
    dx_right = float(x_core[-1] - x_core[-2])
    x = np.concatenate(
        [
            x_core[0] - np.arange(n_pml, 0, -1) * dx_left,
            x_core,
            x_core[-1] + np.arange(1, n_pml + 1) * dx_right,
        ]
    )

    dy_bot = float(y_core[1] - y_core[0])
    dy_top = float(y_core[-1] - y_core[-2])
    y = np.concatenate(
        [
            y_core[0] - np.arange(n_pml, 0, -1) * dy_bot,
            y_core,
            y_core[-1] + np.arange(1, n_pml + 1) * dy_top,
        ]
    )

    dx_min = float(min(np.min(np.diff(x)), np.min(np.diff(y))))
    lz = 10.0 * dx_min
    z = np.array([-0.5 * lz, 0.5 * lz], dtype=float)

    c0_slit = float(centers[-1])
    slit_mask = (x_core >= c0_slit - 0.5 * slit_width - 1e-15) & (
        x_core <= c0_slit + 0.5 * slit_width + 1e-15
    )
    cells_across_slit = int(max(np.count_nonzero(slit_mask) - 1, 0))
    if n > 1:
        b_lo = float(centers[0] + 0.5 * slit_width)
        b_hi = float(centers[1] - 0.5 * slit_width)
        b_mask = (x_core >= b_lo - 1e-15) & (x_core <= b_hi + 1e-15)
        cells_across_bridge = int(max(np.count_nonzero(b_mask) - 1, 0))
    else:
        cells_across_bridge = 0
    scr_mask = (y_core >= -half_t - 1e-15) & (y_core <= half_t + 1e-15)
    cells_across_screen = int(max(np.count_nonzero(scr_mask) - 1, 0))

    return {
        "x": x,
        "y": y,
        "z": z,
        "x_core": x_core,
        "y_core": y_core,
        "lz": lz,
        "centers": centers,
        "dx_base": dx_base,
        "dx_slit": dx_slit,
        "dy_screen": dy_screen,
        "n_pml": int(n_pml),
        "cells_across_slit": cells_across_slit,
        "cells_across_bridge": cells_across_bridge,
        "cells_across_screen": cells_across_screen,
    }


def _compute_2d_farfield_from_n2f(
    n2f: Near2FarMonitor,
    theta: np.ndarray,
    wavelength: float,
    lz: float,
    ifreq: int = 0,
) -> np.ndarray:
    """Convert Wakis 3D Near2FarMonitor radiation intensity to 2D cylindrical intensity."""
    k0 = 2.0 * np.pi / wavelength
    theta_arr = np.asarray(theta, dtype=float)
    theta_sph = np.full_like(theta_arr, 0.5 * np.pi)
    phi_sph = 0.5 * np.pi - theta_arr
    u3d = np.asarray(n2f.radiation_intensity(theta_sph, phi_sph)[ifreq], dtype=float)
    return (2.0 * np.pi / (k0 * (lz**2))) * u3d


def run_n_slit_simulation(
    n_slits: int,
    slit_width: float,
    period: float,
    wavelength: float = DEFAULT_WAVELENGTH,
    extra_wavelengths: tuple[float, ...] = (),
    screen_thickness: float = 0.15 * DEFAULT_WAVELENGTH,
    polarization: str = "Ez",
    with_screen: bool = True,
    cpw: int = 22,
    min_slit_cells: int = 6,
    min_barrier_cells: int = 4,
    min_screen_cells: int = 4,
    x_half_span: float | None = None,
    y_upstream: float = 1.2 * DEFAULT_WAVELENGTH,
    y_downstream: float = 2.2 * DEFAULT_WAVELENGTH,
    record_2d_dft: bool = False,
    theta_deg: np.ndarray | None = None,
    n_cycles: float = 13.0,
) -> dict:
    """Run a 2D FIT N-slit diffraction or empty-domain normalization simulation."""
    f0 = c0 / wavelength
    all_wavelengths = (float(wavelength),) + tuple(float(w) for w in extra_wavelengths)
    freqs = np.asarray([c0 / wl for wl in all_wavelengths], dtype=float)

    grid_info = build_n_slit_grid(
        n_slits=n_slits,
        slit_width=slit_width,
        period=period,
        wavelength=wavelength,
        screen_thickness=screen_thickness,
        x_half_span=x_half_span,
        y_upstream=y_upstream,
        y_downstream=y_downstream,
        cpw=cpw,
        min_slit_cells=min_slit_cells,
        min_barrier_cells=min_barrier_cells,
        min_screen_cells=min_screen_cells,
    )
    x = grid_info["x"]
    y = grid_info["y"]
    z = grid_info["z"]
    x_core = grid_info["x_core"]
    y_core = grid_info["y_core"]
    lz = float(grid_info["lz"])
    centers = grid_info["centers"]
    n_pml = int(grid_info["n_pml"])
    half_t = 0.5 * screen_thickness

    pol = polarization.strip()
    if pol not in ("Ez", "Hz"):
        raise ValueError(f"polarization must be 'Ez' or 'Hz', got {polarization!r}.")
    src_component = "z" if pol == "Ez" else "x"
    dft_component = "Ez" if pol == "Ez" else "Hz"

    boxes = []
    if with_screen:
        left_end = float(x[0] - 2.0 * wavelength)
        right_end = float(x[-1] + 2.0 * wavelength)
        first_left_edge = float(centers[0] - 0.5 * slit_width)
        last_right_edge = float(centers[-1] + 0.5 * slit_width)

        w_left = first_left_edge - left_end
        xc_left = 0.5 * (left_end + first_left_edge)
        boxes.append(Box((xc_left, 0.0, 0.0), (w_left, screen_thickness, 2.0 * lz), "pec"))

        for k in range(int(n_slits) - 1):
            b_lo = float(centers[k] + 0.5 * slit_width)
            b_hi = float(centers[k + 1] - 0.5 * slit_width)
            boxes.append(
                Box(
                    (0.5 * (b_lo + b_hi), 0.0, 0.0),
                    (b_hi - b_lo, screen_thickness, 2.0 * lz),
                    "pec",
                )
            )

        w_right = right_end - last_right_edge
        xc_right = 0.5 * (last_right_edge + right_end)
        boxes.append(Box((xc_right, 0.0, 0.0), (w_right, screen_thickness, 2.0 * lz), "pec"))

    t0 = time.perf_counter()
    grid = GridFIT3D(x=x, y=y, z=z, use_mpi=False, verbose=0)
    solver = SolverFIT3D(
        grid,
        bc_low=["cpml", "cpml", "periodic"],
        bc_high=["cpml", "cpml", "periodic"],
        cfln=0.65,
        n_pml=n_pml,
        verbose=0,
    )
    if boxes:
        assign_materials(solver, boxes, f0)

    y_src = float(y_core[0] + 0.35 * wavelength)
    src_span_x = float(x[-1] - x[0])
    pulse = GaussianSource(frequency=f0, fwidth=0.45 * f0, cutoff=3.5)
    source = CurrentSource(
        src_time=pulse,
        component=src_component,
        center=(0.0, y_src, 0.0),
        size=(src_span_x, 0.0, 2.0 * lz),
        amplitude=1.0,
    )

    w_ref = float((x_core[-1] - x_core[0]) - 0.4 * wavelength)
    y_n2f = float(half_t + 2.5 * grid_info["dy_screen"])

    monitors = []
    flux_wide = FluxMonitor(
        frequencies=freqs,
        center=(0.0, y_n2f, 0.0),
        size=(w_ref, 0.0, 2.0 * lz),
    )
    monitors.append(flux_wide)

    n2f = Near2FarMonitor(
        frequencies=freqs,
        center=(0.0, y_n2f, 0.0),
        size=(w_ref, 0.0, 2.0 * lz),
    )
    monitors.append(n2f)

    dft_2d = None
    if record_2d_dft:
        y_span_core = float(y_core[-1] - y_core[0])
        y_mid_core = 0.5 * float(y_core[-1] + y_core[0])
        dft_2d = DFTMonitor(
            frequencies=freqs[:1],
            components=(dft_component,),
            center=(0.0, y_mid_core, 0.0),
            size=(w_ref, y_span_core, 0.0),
        )
        monitors.append(dft_2d)

    domain_diag = float(np.hypot(x_core[-1] - x_core[0], y_core[-1] - y_core[0]))
    t_total = pulse.t0 + (n_cycles / f0) + (domain_diag / c0)
    n_steps = int(np.ceil(t_total / solver.dt))

    for step in range(n_steps):
        t = step * solver.dt
        source.update(solver, t)
        solver.one_step()
        for mon in monitors:
            mon.update(solver, t)
    elapsed_s = time.perf_counter() - t0

    if theta_deg is None:
        theta_deg = np.linspace(-82.0, 82.0, 1641)
    theta_rad = np.deg2rad(theta_deg)

    s_freqs = pulse.fourier_transform(freqs)
    power_norms = 2.0 * (np.abs(s_freqs) ** 2)

    u2d_all = []
    for ifreq, wl in enumerate(all_wavelengths):
        u_i = _compute_2d_farfield_from_n2f(n2f, theta_rad, wl, lz, ifreq=ifreq) / float(
            power_norms[ifreq]
        )
        u2d_all.append(u_i)
    u2d_all_arr = np.asarray(u2d_all)
    u2d = u2d_all_arr[0]
    u2d_integral = float(np.trapezoid(u2d, theta_rad))

    p_wide_per_z = float(flux_wide.get_flux()[0] / (lz * float(power_norms[0])))
    p_n2f_per_z = float(n2f.get_flux()[0] / (lz * float(power_norms[0])))

    est = estimate(
        cells=(grid.Nx, grid.Ny, grid.Nz),
        steps=n_steps,
        cpml=True,
        monitors=[(freqs.size, grid.Nx, 2)],
    )

    map_2d = None
    if dft_2d is not None:
        fld_2d, coords_2d = dft_2d.get_field(dft_component)
        map_2d = {
            "x": coords_2d[0].copy(),
            "y": coords_2d[1].copy(),
            "field": (np.squeeze(fld_2d[0]) / complex(s_freqs[0])).copy(),
        }

    return {
        "n_slits": int(n_slits),
        "slit_width": float(slit_width),
        "period": float(period),
        "wavelength": float(wavelength),
        "all_wavelengths": np.asarray(all_wavelengths),
        "cpw": int(cpw),
        "nx": int(grid.Nx),
        "ny": int(grid.Ny),
        "nz": int(grid.Nz),
        "n_cells": int(grid.Nx * grid.Ny * grid.Nz),
        "n_steps": int(n_steps),
        "dt": float(solver.dt),
        "elapsed_s": float(elapsed_s),
        "host_ram_mb": float(est.host_bytes / 1e6),
        "cells_across_slit": int(grid_info["cells_across_slit"]),
        "cells_across_bridge": int(grid_info["cells_across_bridge"]),
        "cells_across_screen": int(grid_info["cells_across_screen"]),
        "w_ref": w_ref,
        "p_wide_per_z": p_wide_per_z,
        "p_n2f_per_z": p_n2f_per_z,
        "u2d_integral": u2d_integral,
        "parseval_ratio": u2d_integral / max(abs(p_n2f_per_z), 1e-30),
        "theta_deg": theta_deg.copy(),
        "theta_rad": theta_rad.copy(),
        "u2d": u2d.copy(),
        "u2d_all": u2d_all_arr.copy(),
        "map_2d": map_2d,
    }


def run_periodic_unit_cell_grating(
    slit_width: float,
    period: float,
    wavelength: float = DEFAULT_WAVELENGTH,
    screen_thickness: float = 0.15 * DEFAULT_WAVELENGTH,
    polarization: str = "Ez",
    cpw: int = 24,
    min_slit_cells: int = 8,
    min_barrier_cells: int = 8,
    min_screen_cells: int = 4,
    y_upstream: float = 1.4 * DEFAULT_WAVELENGTH,
    y_downstream: float = 1.8 * DEFAULT_WAVELENGTH,
    n_pml: int = 10,
    n_cycles: float = 18.0,
) -> dict:
    """Solve the N -> infinity periodic grating in one period d with periodic BCs & DiffractionMonitor.

    In a single periodic unit cell x in [-d/2, +d/2] with ``bc_low=['periodic', 'cpml',
    'periodic']``, the slit occupies [-a/2, +a/2] and the PEC screen bars occupy
    [-d/2, -a/2] and [+a/2, +d/2]. ``wakis.diffraction.DiffractionMonitor`` decomposes
    the downstream transmitted field into exact Rayleigh-Floquet plane-wave harmonics
    k_{x,m} = 2*pi*m / d and returns their diffraction efficiencies T_m.
    """
    f0 = c0 / wavelength
    freqs = np.array([f0])
    dx_base = wavelength / float(cpw)
    bridge_width = period - slit_width
    dx_slit = min(dx_base, slit_width / float(max(min_slit_cells, 4)))
    dx_bar = min(dx_base, 0.5 * bridge_width / float(max(min_barrier_cells // 2, 2)))
    dy_screen = min(dx_base, screen_thickness / float(max(min_screen_cells, 4)))

    # DiffractionMonitor requires a uniform mesh along the periodic transverse axis x.
    # Since period / slit_width = 3, choosing n_x as a multiple of 6 snaps exact uniform
    # nodes to x = -d/2, -a/2, 0, +a/2, +d/2 simultaneously.
    n_x = max(24, 6 * int(round(cpw * (period / wavelength) / 6.0)))
    x = np.linspace(-0.5 * period, 0.5 * period, n_x + 1)
    dx_bar = float(x[1] - x[0])

    half_t = 0.5 * screen_thickness
    y_core = graded_axis(
        [-y_upstream, -half_t, half_t, y_downstream],
        [dx_base, dy_screen, dx_base],
        min_cells=[6, max(min_screen_cells, 4), 8],
        grading=1.35,
    )
    dy_bot = float(y_core[1] - y_core[0])
    dy_top = float(y_core[-1] - y_core[-2])
    y = np.concatenate(
        [
            y_core[0] - np.arange(n_pml, 0, -1) * dy_bot,
            y_core,
            y_core[-1] + np.arange(1, n_pml + 1) * dy_top,
        ]
    )

    dx_min = float(min(np.min(np.diff(x)), np.min(np.diff(y))))
    lz = 10.0 * dx_min
    z = np.array([-0.5 * lz, 0.5 * lz], dtype=float)

    pol = polarization.strip()
    src_component = "z" if pol == "Ez" else "x"

    y_src = float(y_core[0] + 0.35 * wavelength)
    # Place the DiffractionMonitor near the top of y_core where dy is uniform (= dy_top)
    y_mon = float(y_core[-3])

    def _run_cell(with_grating: bool) -> tuple[DiffractionMonitor, GridFIT3D, int, float]:
        t0 = time.perf_counter()
        grid = GridFIT3D(x=x, y=y, z=z, use_mpi=False, verbose=0)
        solver = SolverFIT3D(
            grid,
            bc_low=["periodic", "cpml", "periodic"],
            bc_high=["periodic", "cpml", "periodic"],
            cfln=0.65,
            n_pml=n_pml,
            verbose=0,
        )
        if with_grating:
            w_bar = 0.5 * (period - slit_width) + 2.0 * dx_bar
            xc_left = -0.5 * slit_width - 0.5 * w_bar
            xc_right = 0.5 * slit_width + 0.5 * w_bar
            boxes = [
                Box((xc_left, 0.0, 0.0), (w_bar, screen_thickness, 2.0 * lz), "pec"),
                Box((xc_right, 0.0, 0.0), (w_bar, screen_thickness, 2.0 * lz), "pec"),
            ]
            assign_materials(solver, boxes, f0)

        pulse = GaussianSource(frequency=f0, fwidth=0.45 * f0, cutoff=3.5)
        source = CurrentSource(
            src_time=pulse,
            component=src_component,
            center=(0.0, y_src, 0.0),
            size=(2.0 * period, 0.0, 2.0 * lz),
            amplitude=1.0,
        )
        diff_mon = DiffractionMonitor(
            frequencies=freqs,
            center=(0.0, y_mon, 0.0),
            size=(period, 0.0, lz),
        )
        t_total = pulse.t0 + (n_cycles / f0) + ((y_core[-1] - y_core[0]) / c0)
        n_steps = int(np.ceil(t_total / solver.dt))
        for step in range(n_steps):
            t = step * solver.dt
            source.update(solver, t)
            solver.one_step()
            diff_mon.update(solver, t)
        elapsed = time.perf_counter() - t0
        return diff_mon, grid, n_steps, elapsed

    mon_inc, _, _, _ = _run_cell(with_grating=False)
    mon_grat, grid_g, n_steps_g, elapsed_g = _run_cell(with_grating=True)

    power_dict = mon_grat.get_power(incident=mon_inc.get_flux())
    orders_1d = np.asarray(mon_grat.orders[1], dtype=int)
    eta_orders = np.asarray(power_dict["up"][0, 0, :], dtype=float)
    theta_mag_rad = np.asarray(mon_grat.order_angles()[0][0, 0, :], dtype=float)
    theta_orders_deg = np.rad2deg(np.sign(orders_1d) * theta_mag_rad)
    total_trans = float(np.nansum(eta_orders))
    order_fractions = eta_orders / max(total_trans, 1e-30)

    est = estimate(
        cells=(grid_g.Nx, grid_g.Ny, grid_g.Nz),
        steps=n_steps_g,
        cpml=True,
        monitors=[(1, grid_g.Nx, 4)],
    )

    return {
        "orders": orders_1d,
        "theta_orders_deg": theta_orders_deg,
        "efficiencies": eta_orders,
        "order_fractions": order_fractions,
        "total_transmission": total_trans,
        "n_cells": int(grid_g.Nx * grid_g.Ny * grid_g.Nz),
        "n_steps": int(n_steps_g),
        "elapsed_s": float(elapsed_g),
        "host_ram_mb": float(est.host_bytes / 1e6),
    }


def run_preflight_summary(
    wavelength: float = DEFAULT_WAVELENGTH,
    n_slits: int = 6,
    slit_width: float = 0.6 * DEFAULT_WAVELENGTH,
    period: float = 1.8 * DEFAULT_WAVELENGTH,
    screen_thickness: float = 0.15 * DEFAULT_WAVELENGTH,
    cpw: int = 22,
) -> dict:
    """Run ``wakis.estimate.estimate`` and ``preflight`` on the N = 6 grating mesh."""
    grid_info = build_n_slit_grid(
        n_slits=n_slits,
        slit_width=slit_width,
        period=period,
        wavelength=wavelength,
        screen_thickness=screen_thickness,
        cpw=cpw,
    )
    grid = GridFIT3D(
        x=grid_info["x"],
        y=grid_info["y"],
        z=grid_info["z"],
        use_mpi=False,
        verbose=0,
    )
    dx_min = float(min(np.min(grid.dx), np.min(grid.dy)))
    dt_est = 0.65 * dx_min / (c0 * np.sqrt(2.0))
    est = estimate(
        cells=(grid.Nx, grid.Ny, grid.Nz),
        steps=650,
        cpml=True,
        monitors=[(1, grid.Nx, 1), (1, grid.Nx * grid.Ny, 1)],
    )
    problems = preflight(est, "laptop", raise_error=False)
    return {
        "n_cells": int(est.cells),
        "solver_ram_mb": float(est.breakdown.get("solver (host)", 0.0) / 1e6),
        "monitor_ram_mb": float(est.breakdown.get("monitors", 0.0) / 1e6),
        "total_ram_mb": float(est.host_bytes / 1e6),
        "wall_clock_s": float(est.seconds),
        "dt": dt_est,
        "cells_across_slit": int(grid_info["cells_across_slit"]),
        "cells_across_bridge": int(grid_info["cells_across_bridge"]),
        "cells_across_screen": int(grid_info["cells_across_screen"]),
        "ok": len(problems) == 0,
        "warnings": list(problems),
    }



def _extract_lobe_widths_and_secondary_maxima(
    theta_deg: np.ndarray,
    u2d_norm: np.ndarray,
    n_slits: int,
    period: float,
    wavelength: float,
) -> dict[str, float | int | np.ndarray]:
    """Extract main-lobe FWHM, HWZM, m=+1 peak angle, and (N-2) secondary maxima."""
    theta_rad = np.deg2rad(theta_deg)
    sin_theta = np.sin(theta_rad)
    d_over_lam = period / wavelength

    # 1. Central m = 0 lobe FWHM and first-zero HWZM in sin(theta)
    pos_mask = (sin_theta >= 0.0) & (sin_theta <= 1.6 / (n_slits * d_over_lam))
    sin_pos = sin_theta[pos_mask]
    u_pos = u2d_norm[pos_mask]
    # Half-maximum crossing on the positive side of m = 0
    above_half = np.where(u_pos >= 0.5)[0]
    if above_half.size > 0 and above_half[-1] + 1 < sin_pos.size:
        i0 = int(above_half[-1])
        frac = (0.5 - u_pos[i0]) / max(abs(u_pos[i0 + 1] - u_pos[i0]), 1e-30)
        hwhm_sin = float(sin_pos[i0] + frac * (sin_pos[i0 + 1] - sin_pos[i0]))
    else:
        hwhm_sin = float(0.44295 / (n_slits * d_over_lam))
    fwhm_sin = 2.0 * hwhm_sin

    # First minimum (HWZM) near sin(theta) = lambda / (N * d)
    zero_win = (sin_theta >= 0.65 / (n_slits * d_over_lam)) & (
        sin_theta <= 1.35 / (n_slits * d_over_lam)
    )
    if np.any(zero_win):
        imin = int(np.argmin(u2d_norm[zero_win]))
        hwzm_sin = float(sin_theta[zero_win][imin])
    else:
        hwzm_sin = float(1.0 / (n_slits * d_over_lam))

    # 2. First principal maximum m = +1 near sin(theta) = lambda / d
    target_sin1 = 1.0 / d_over_lam
    m1_win = (sin_theta >= target_sin1 - 0.8 / (n_slits * d_over_lam)) & (
        sin_theta <= target_sin1 + 0.8 / (n_slits * d_over_lam)
    )
    imax1 = int(np.argmax(u2d_norm[m1_win]))
    theta_m1_deg = float(theta_deg[m1_win][imax1])
    sin_m1 = float(sin_theta[m1_win][imax1])

    # 3. Count local maxima (secondary maxima) strictly between the 1st and (N-1)-th zeros:
    # sin(theta) in [1.15 / (N * d/lam), (N - 1.15) / (N * d/lam)]
    sec_peaks_sin = []
    sec_peaks_val = []
    if n_slits >= 3:
        s_low = 1.12 / (n_slits * d_over_lam)
        s_high = (n_slits - 1.12) / (n_slits * d_over_lam)
        sec_mask = (sin_theta >= s_low) & (sin_theta <= s_high)
        s_sec = sin_theta[sec_mask]
        u_sec = u2d_norm[sec_mask]
        # Smooth slightly over 3 points to avoid grid jitter false peaks, then find local maxima
        if u_sec.size >= 5:
            u_smooth = np.convolve(u_sec, np.ones(3) / 3.0, mode="same")
            for k in range(1, u_smooth.size - 1):
                if u_smooth[k] > u_smooth[k - 1] and u_smooth[k] >= u_smooth[k + 1]:
                    if not sec_peaks_sin or (
                        s_sec[k] - sec_peaks_sin[-1] > 0.45 / (n_slits * d_over_lam)
                    ):
                        sec_peaks_sin.append(float(s_sec[k]))
                        sec_peaks_val.append(float(u_sec[k]))

    max_sec_val = float(max(sec_peaks_val)) if sec_peaks_val else 0.0
    return {
        "fwhm_sin": fwhm_sin,
        "hwzm_sin": hwzm_sin,
        "theta_m1_deg": theta_m1_deg,
        "sin_m1": sin_m1,
        "n_secondary_counted": int(len(sec_peaks_val)),
        "max_secondary_rel": max_sec_val,
        "sec_peaks_sin": np.asarray(sec_peaks_sin, dtype=float),
        "sec_peaks_val": np.asarray(sec_peaks_val, dtype=float),
    }


def run_patterns_task(quick: bool = False) -> dict:
    """Run N-slit diffraction patterns for N in {2, 3, 4, 6, 8}, secondary maxima, and 2D field map."""
    wavelength = DEFAULT_WAVELENGTH
    slit_width = 0.6 * wavelength
    period = 1.8 * wavelength
    screen_thickness = 0.14 * wavelength
    cpw = 20 if quick else 22
    n_list = [2, 3, 4, 6] if quick else [2, 3, 4, 6, 8]

    theta_deg = np.linspace(-80.0, 80.0, 1601)
    theta_rad = np.deg2rad(theta_deg)
    idx0 = int(np.argmin(np.abs(theta_deg)))
    ang_mask = np.abs(theta_deg) <= 68.0

    # Empty-domain reference for the N = 6 near-field intensity map
    n_map_slits = 6
    empty_ref = run_n_slit_simulation(
        n_slits=n_map_slits,
        slit_width=slit_width,
        period=period,
        wavelength=wavelength,
        screen_thickness=screen_thickness,
        polarization="Hz",
        with_screen=False,
        cpw=cpw,
        y_upstream=1.2 * wavelength,
        y_downstream=4.2 * wavelength,
        record_2d_dft=True,
        theta_deg=theta_deg,
        n_cycles=11.5 if quick else 13.5,
    )
    h_inc_rms = float(
        np.mean(
            np.abs(
                empty_ref["map_2d"]["field"][
                    np.abs(empty_ref["map_2d"]["x"]) <= 3.0 * wavelength, :
                ]
            )
        )
    )

    single_env_theory = single_slit_envelope(theta_rad, slit_width, wavelength, obliquity="rs2")

    patterns_fit = []
    patterns_theory = []
    l2_errors = []
    fwhm_sin_fit = []
    fwhm_sin_theory = []
    hwzm_sin_fit = []
    hwzm_sin_theory = []
    theta_m1_fit = []
    theta_m1_theory = []
    n_sec_counted = []
    n_sec_expected = []
    max_sec_fit = []
    max_sec_theory = []
    n_cells_list = []
    elapsed_s_list = []
    host_ram_mb_list = []
    lobe_fractions_m0_m1 = []

    map_x_over_lam = None
    map_y_over_lam = None
    map_intensity = None

    for n_slits in n_list:
        rec_2d = n_slits == n_map_slits
        sim = run_n_slit_simulation(
            n_slits=n_slits,
            slit_width=slit_width,
            period=period,
            wavelength=wavelength,
            screen_thickness=screen_thickness,
            polarization="Hz",
            with_screen=True,
            cpw=cpw,
            y_upstream=1.2 * wavelength,
            y_downstream=4.2 * wavelength if rec_2d else 2.0 * wavelength,
            record_2d_dft=rec_2d,
            theta_deg=theta_deg,
            n_cycles=11.5 if quick else 13.5,
        )
        pat_fit = sim["u2d"] / float(sim["u2d"][idx0])
        pat_th = n_slit_farfield_pattern(
            theta_rad, n_slits, slit_width, period, wavelength, obliquity="rs2"
        )
        err = float(
            np.linalg.norm(pat_fit[ang_mask] - pat_th[ang_mask])
            / np.linalg.norm(pat_th[ang_mask])
        )
        metrics_fit = _extract_lobe_widths_and_secondary_maxima(
            theta_deg, pat_fit, n_slits, period, wavelength
        )
        metrics_th = _extract_lobe_widths_and_secondary_maxima(
            theta_deg, pat_th, n_slits, period, wavelength
        )

        # Integrate angular power inside principal orders m = -1, 0, +1
        sin_th = np.sin(theta_rad)
        half_win = 0.5 * (wavelength / period)
        total_int = float(np.trapezoid(sim["u2d"], theta_rad))
        fracs = []
        for m_ord in (-1, 0, 1):
            s_center = m_ord * (wavelength / period)
            mask_m = (sin_th >= s_center - half_win) & (sin_th <= s_center + half_win)
            fracs.append(float(np.trapezoid(sim["u2d"][mask_m], theta_rad[mask_m]) / total_int))
        lobe_fractions_m0_m1.append(fracs)

        patterns_fit.append(pat_fit)
        patterns_theory.append(pat_th)
        l2_errors.append(err)
        fwhm_sin_fit.append(metrics_fit["fwhm_sin"])
        fwhm_sin_theory.append(principal_peak_fwhm_sin_theta(n_slits, period, wavelength))
        hwzm_sin_fit.append(metrics_fit["hwzm_sin"])
        hwzm_sin_theory.append(principal_peak_hwzm_sin_theta(n_slits, period, wavelength))
        theta_m1_fit.append(metrics_fit["theta_m1_deg"])
        theta_m1_theory.append(float(np.rad2deg(np.arcsin(wavelength / period))))
        n_sec_counted.append(metrics_fit["n_secondary_counted"])
        n_sec_expected.append(secondary_maxima_count(n_slits))
        max_sec_fit.append(metrics_fit["max_secondary_rel"])
        max_sec_theory.append(metrics_th["max_secondary_rel"])
        n_cells_list.append(sim["n_cells"])
        elapsed_s_list.append(sim["elapsed_s"])
        host_ram_mb_list.append(sim["host_ram_mb"])

        if rec_2d:
            map_x_over_lam = sim["map_2d"]["x"] / wavelength
            map_y_over_lam = sim["map_2d"]["y"] / wavelength
            map_intensity = (np.abs(sim["map_2d"]["field"]) / h_inc_rms) ** 2

    return {
        "slit_width_over_lam": np.array(slit_width / wavelength),
        "period_over_lam": np.array(period / wavelength),
        "n_values": np.asarray(n_list, dtype=int),
        "theta_deg": theta_deg,
        "single_env_theory": single_env_theory,
        "patterns_fit": np.asarray(patterns_fit),
        "patterns_theory": np.asarray(patterns_theory),
        "l2_errors": np.asarray(l2_errors),
        "fwhm_sin_fit": np.asarray(fwhm_sin_fit),
        "fwhm_sin_theory": np.asarray(fwhm_sin_theory),
        "hwzm_sin_fit": np.asarray(hwzm_sin_fit),
        "hwzm_sin_theory": np.asarray(hwzm_sin_theory),
        "theta_m1_fit": np.asarray(theta_m1_fit),
        "theta_m1_theory": np.asarray(theta_m1_theory),
        "n_sec_counted": np.asarray(n_sec_counted, dtype=int),
        "n_sec_expected": np.asarray(n_sec_expected, dtype=int),
        "max_sec_fit": np.asarray(max_sec_fit),
        "max_sec_theory": np.asarray(max_sec_theory),
        "lobe_fractions": np.asarray(lobe_fractions_m0_m1),
        "n_cells": np.asarray(n_cells_list, dtype=int),
        "elapsed_s": np.asarray(elapsed_s_list),
        "host_ram_mb": np.asarray(host_ram_mb_list),
        "map_x_over_lam": map_x_over_lam,
        "map_y_over_lam": map_y_over_lam,
        "map_intensity": map_intensity,
    }


def run_resolving_power_task(quick: bool = False) -> dict:
    """Demonstrate Rayleigh chromatic resolving power R = lambda / Delta_lambda = |m| N."""
    wavelength = DEFAULT_WAVELENGTH
    n_slits = 6
    order_m = 1
    slit_width = 0.6 * wavelength
    period = 1.8 * wavelength
    screen_thickness = 0.14 * wavelength
    cpw = 20 if quick else 22

    r_rayleigh = float(rayleigh_resolving_power(n_slits, order_m))  # R = 6
    # Three test wavelength shifts Delta_lambda / lambda_0 relative to 1 / (m * N):
    # - sub-Rayleigh: 0.55 / R (unresolved single peak)
    # - exact Rayleigh: 1.00 / R (peak of lambda_2 at first zero of lambda_1; ~81% dip)
    # - super-Rayleigh: 1.45 / R (well-resolved doublet)
    frac_shifts = np.array([0.55 / r_rayleigh, 1.00 / r_rayleigh, 1.45 / r_rayleigh])
    extra_wls = [float(wavelength * (1.0 + df)) for df in frac_shifts]

    theta_deg = np.linspace(15.0, 58.0, 861)
    theta_rad = np.deg2rad(theta_deg)

    sim = run_n_slit_simulation(
        n_slits=n_slits,
        slit_width=slit_width,
        period=period,
        wavelength=wavelength,
        extra_wavelengths=extra_wls,
        screen_thickness=screen_thickness,
        polarization="Hz",
        with_screen=True,
        cpw=cpw,
        y_upstream=1.2 * wavelength,
        y_downstream=2.0 * wavelength,
        record_2d_dft=False,
        theta_deg=theta_deg,
        n_cycles=13.0 if quick else 15.0,
    )

    u_all = sim["u2d_all"]  # shape (4, n_theta)
    # Normalize each monochromatic line to unit peak in order m = +1 so we compare an
    # equal-intensity spectral doublet at (lambda_0, lambda_0 + Delta_lambda)
    u_base_fit = u_all[0] / float(np.max(u_all[0]))
    u_base_th = n_slit_farfield_pattern(
        theta_rad, n_slits, slit_width, period, wavelength, obliquity="rs2"
    )
    u_base_th = u_base_th / float(np.max(u_base_th))

    u_shifted_fit = []
    u_shifted_th = []
    doublet_sum_fit = []
    doublet_sum_th = []
    saddle_to_peak_fit = []
    saddle_to_peak_th = []

    for k, wl_2 in enumerate(extra_wls):
        u2_fit = u_all[k + 1] / float(np.max(u_all[k + 1]))
        u2_th = n_slit_farfield_pattern(
            theta_rad, n_slits, slit_width, period, wl_2, obliquity="rs2"
        )
        u2_th = u2_th / float(np.max(u2_th))

        tot_fit = u_base_fit + u2_fit
        tot_th = u_base_th + u2_th

        idx1 = int(np.argmax(u_base_fit))
        idx2 = int(np.argmax(u2_fit))
        lo, hi = min(idx1, idx2), max(idx1, idx2)
        if hi > lo + 2:
            saddle_fit = float(np.min(tot_fit[lo : hi + 1]))
            peak_ref_fit = float(0.5 * (tot_fit[lo] + tot_fit[hi]))
            ratio_fit = saddle_fit / max(peak_ref_fit, 1e-30)
        else:
            ratio_fit = 1.0

        idx1_t = int(np.argmax(u_base_th))
        idx2_t = int(np.argmax(u2_th))
        lo_t, hi_t = min(idx1_t, idx2_t), max(idx1_t, idx2_t)
        if hi_t > lo_t + 2:
            saddle_th = float(np.min(tot_th[lo_t : hi_t + 1]))
            peak_ref_th = float(0.5 * (tot_th[lo_t] + tot_th[hi_t]))
            ratio_th = saddle_th / max(peak_ref_th, 1e-30)
        else:
            ratio_th = 1.0

        u_shifted_fit.append(u2_fit)
        u_shifted_th.append(u2_th)
        doublet_sum_fit.append(tot_fit)
        doublet_sum_th.append(tot_th)
        saddle_to_peak_fit.append(ratio_fit)
        saddle_to_peak_th.append(ratio_th)

    return {
        "n_slits": np.array(n_slits),
        "order_m": np.array(order_m),
        "resolving_power": np.array(r_rayleigh),
        "frac_shifts": frac_shifts,
        "theta_deg": theta_deg,
        "u_base_fit": u_base_fit,
        "u_base_th": u_base_th,
        "u_shifted_fit": np.asarray(u_shifted_fit),
        "u_shifted_th": np.asarray(u_shifted_th),
        "doublet_sum_fit": np.asarray(doublet_sum_fit),
        "doublet_sum_th": np.asarray(doublet_sum_th),
        "saddle_to_peak_fit": np.asarray(saddle_to_peak_fit),
        "saddle_to_peak_th": np.asarray(saddle_to_peak_th),
    }


def run_periodic_vs_finite_task(quick: bool = False) -> dict:
    """Compare N -> infinity periodic unit cell (DiffractionMonitor) vs finite-N cost and efficiencies."""
    wavelength = DEFAULT_WAVELENGTH
    slit_width = 0.6 * wavelength
    period = 1.8 * wavelength
    screen_thickness = 0.14 * wavelength
    cpw = 20 if quick else 22

    per_res = run_periodic_unit_cell_grating(
        slit_width=slit_width,
        period=period,
        wavelength=wavelength,
        screen_thickness=screen_thickness,
        polarization="Hz",
        cpw=cpw,
        n_cycles=15.0 if quick else 18.0,
    )

    # Extract m in {-1, 0, +1} from DiffractionMonitor
    orders_all = per_res["orders"]
    eta_all = per_res["efficiencies"]
    frac_all = per_res["order_fractions"]
    theta_all = per_res["theta_orders_deg"]

    target_orders = np.array([-1, 0, 1], dtype=int)
    eta_m = []
    frac_m = []
    theta_m = []
    for m_val in target_orders:
        idx = np.where(orders_all == m_val)[0]
        if idx.size > 0:
            i0 = int(idx[0])
            eta_m.append(float(eta_all[i0]))
            frac_m.append(float(frac_all[i0]))
            theta_m.append(float(theta_all[i0]))
        else:
            eta_m.append(0.0)
            frac_m.append(0.0)
            theta_m.append(float(np.rad2deg(np.arcsin(m_val * wavelength / period))))

    # Projected cost for large finite N using build_n_slit_grid + wakis.estimate.estimate
    n_proj = np.array([2, 3, 4, 6, 8, 12, 16, 24, 32], dtype=int)
    proj_cells = []
    proj_ram_mb = []
    proj_seconds = []
    for n_val in n_proj:
        g_info = build_n_slit_grid(
            n_slits=int(n_val),
            slit_width=slit_width,
            period=period,
            wavelength=wavelength,
            screen_thickness=screen_thickness,
            cpw=cpw,
        )
        nx = g_info["x"].size - 1
        ny = g_info["y"].size - 1
        nz = 1
        diag = float(
            np.hypot(
                g_info["x_core"][-1] - g_info["x_core"][0],
                g_info["y_core"][-1] - g_info["y_core"][0],
            )
        )
        dx_min = float(min(np.min(np.diff(g_info["x"])), np.min(np.diff(g_info["y"]))))
        dt_e = 0.65 * dx_min / (c0 * np.sqrt(2.0))
        t_tot = (16.0 * wavelength / c0) + (diag / c0)
        steps_e = int(np.ceil(t_tot / dt_e))
        est_n = estimate(
            cells=(nx, ny, nz),
            steps=steps_e,
            cpml=True,
            monitors=[(1, nx, 2)],
        )
        proj_cells.append(int(est_n.cells))
        proj_ram_mb.append(float(est_n.host_bytes / 1e6))
        proj_seconds.append(float(est_n.seconds))

    return {
        "target_orders": target_orders,
        "theta_orders_deg": np.asarray(theta_m),
        "periodic_efficiencies": np.asarray(eta_m),
        "periodic_order_fractions": np.asarray(frac_m),
        "periodic_total_transmission": np.array(per_res["total_transmission"]),
        "periodic_n_cells": np.array(per_res["n_cells"]),
        "periodic_n_steps": np.array(per_res["n_steps"]),
        "periodic_elapsed_s": np.array(per_res["elapsed_s"]),
        "periodic_host_ram_mb": np.array(per_res["host_ram_mb"]),
        "n_proj": n_proj,
        "proj_cells": np.asarray(proj_cells, dtype=int),
        "proj_ram_mb": np.asarray(proj_ram_mb),
        "proj_seconds": np.asarray(proj_seconds),
    }


def run_resolution_task(quick: bool = False) -> dict:
    """Grid convergence study over cpw for an N = 4 grating (a = 0.6*lambda, d = 1.8*lambda)."""
    wavelength = DEFAULT_WAVELENGTH
    n_slits = 4
    slit_width = 0.6 * wavelength
    period = 1.8 * wavelength
    screen_thickness = 0.14 * wavelength
    cpw_list = [14, 20, 28] if quick else [12, 16, 20, 28, 36]

    theta_deg = np.linspace(-80.0, 80.0, 801)
    theta_rad = np.deg2rad(theta_deg)
    idx0 = int(np.argmin(np.abs(theta_deg)))
    ang_mask = np.abs(theta_deg) <= 65.0
    theory_ref = n_slit_farfield_pattern(
        theta_rad, n_slits, slit_width, period, wavelength, obliquity="rs2"
    )

    l2_vs_theory = []
    patterns = []
    cells_slit_list = []
    cells_bridge_list = []
    cells_screen_list = []
    elapsed_list = []

    for cpw in cpw_list:
        sc = max(4, int(round(0.6 * cpw)))
        bc = max(4, int(round(1.2 * cpw)))
        tc = max(3, int(round(0.14 * cpw)))
        sim = run_n_slit_simulation(
            n_slits=n_slits,
            slit_width=slit_width,
            period=period,
            wavelength=wavelength,
            screen_thickness=screen_thickness,
            polarization="Hz",
            with_screen=True,
            cpw=cpw,
            min_slit_cells=sc,
            min_barrier_cells=bc,
            min_screen_cells=tc,
            y_upstream=1.2 * wavelength,
            y_downstream=2.0 * wavelength,
            record_2d_dft=False,
            theta_deg=theta_deg,
            n_cycles=12.0,
        )
        pat = sim["u2d"] / float(sim["u2d"][idx0])
        err = float(
            np.linalg.norm(pat[ang_mask] - theory_ref[ang_mask])
            / np.linalg.norm(theory_ref[ang_mask])
        )
        l2_vs_theory.append(err)
        patterns.append(pat)
        cells_slit_list.append(sim["cells_across_slit"])
        cells_bridge_list.append(sim["cells_across_bridge"])
        cells_screen_list.append(sim["cells_across_screen"])
        elapsed_list.append(sim["elapsed_s"])

    patterns_arr = np.asarray(patterns)
    rel_to_finest = [
        float(
            np.linalg.norm(p[ang_mask] - patterns_arr[-1][ang_mask])
            / np.linalg.norm(patterns_arr[-1][ang_mask])
        )
        for p in patterns_arr
    ]

    return {
        "cpw_values": np.asarray(cpw_list, dtype=int),
        "theta_deg": theta_deg,
        "patterns": patterns_arr,
        "theory_ref": theory_ref,
        "l2_vs_theory": np.asarray(l2_vs_theory),
        "rel_to_finest": np.asarray(rel_to_finest),
        "cells_across_slit": np.asarray(cells_slit_list, dtype=int),
        "cells_across_bridge": np.asarray(cells_bridge_list, dtype=int),
        "cells_across_screen": np.asarray(cells_screen_list, dtype=int),
        "elapsed_s": np.asarray(elapsed_list),
    }


def _build_finite_n_schematic_solver(
    n_slits: int = 6,
    slit_width: float = 0.6 * DEFAULT_WAVELENGTH,
    period: float = 1.8 * DEFAULT_WAVELENGTH,
    wavelength: float = DEFAULT_WAVELENGTH,
    screen_thickness: float = 0.14 * DEFAULT_WAVELENGTH,
    cpw: int = 22,
    y_upstream: float = 1.2 * DEFAULT_WAVELENGTH,
    y_downstream: float = 3.6 * DEFAULT_WAVELENGTH,
) -> tuple[SolverFIT3D, dict, list[Box], CurrentSource, list]:
    """Build a lightweight 2D finite-N grating solver, PEC boxes, source, and monitors."""
    f0 = c0 / wavelength
    grid_info = build_n_slit_grid(
        n_slits=n_slits,
        slit_width=slit_width,
        period=period,
        wavelength=wavelength,
        screen_thickness=screen_thickness,
        y_upstream=y_upstream,
        y_downstream=y_downstream,
        cpw=cpw,
    )
    x = grid_info["x"]
    y = grid_info["y"]
    z = grid_info["z"]
    x_core = grid_info["x_core"]
    y_core = grid_info["y_core"]
    lz = float(grid_info["lz"])
    centers = grid_info["centers"]
    n_pml = int(grid_info["n_pml"])
    half_t = 0.5 * screen_thickness

    grid = GridFIT3D(x=x, y=y, z=z, use_mpi=False, verbose=0)
    solver = SolverFIT3D(
        grid,
        bc_low=["cpml", "cpml", "periodic"],
        bc_high=["cpml", "cpml", "periodic"],
        cfln=0.65,
        n_pml=n_pml,
        verbose=0,
    )

    left_end = float(x[0])
    right_end = float(x[-1])
    first_left_edge = float(centers[0] - 0.5 * slit_width)
    last_right_edge = float(centers[-1] + 0.5 * slit_width)
    boxes = [
        Box(
            (0.5 * (left_end + first_left_edge), 0.0, 0.0),
            (first_left_edge - left_end, screen_thickness, 2.0 * lz),
            "pec",
        )
    ]
    for k in range(int(n_slits) - 1):
        b_lo = float(centers[k] + 0.5 * slit_width)
        b_hi = float(centers[k + 1] - 0.5 * slit_width)
        boxes.append(
            Box(
                (0.5 * (b_lo + b_hi), 0.0, 0.0),
                (b_hi - b_lo, screen_thickness, 2.0 * lz),
                "pec",
            )
        )
    boxes.append(
        Box(
            (0.5 * (last_right_edge + right_end), 0.0, 0.0),
            (right_end - last_right_edge, screen_thickness, 2.0 * lz),
            "pec",
        )
    )
    for b in boxes:
        b.material = "pec"
    assign_materials(solver, boxes, f0)

    y_src = float(y_core[0] + 0.35 * wavelength)
    src_span_x = float(x[-1] - x[0])
    pulse = GaussianSource(frequency=f0, fwidth=0.45 * f0, cutoff=3.5)
    source = CurrentSource(
        src_time=pulse,
        component="x",
        center=(0.0, y_src, 0.0),
        size=(src_span_x, 0.0, 2.0 * lz),
        amplitude=1.0,
    )

    freqs = np.array([f0])
    w_ref = float((x_core[-1] - x_core[0]) - 0.4 * wavelength)
    y_n2f = float(half_t + 2.5 * grid_info["dy_screen"])
    flux_wide = FluxMonitor(
        frequencies=freqs,
        center=(0.0, y_n2f, 0.0),
        size=(w_ref, 0.0, 2.0 * lz),
    )
    n2f = Near2FarMonitor(
        frequencies=freqs,
        center=(0.0, y_n2f, 0.0),
        size=(w_ref, 0.0, 2.0 * lz),
    )
    return solver, grid_info, boxes, source, [flux_wide, n2f]


def _build_periodic_cell_schematic_solver(
    slit_width: float = 0.6 * DEFAULT_WAVELENGTH,
    period: float = 1.8 * DEFAULT_WAVELENGTH,
    wavelength: float = DEFAULT_WAVELENGTH,
    screen_thickness: float = 0.14 * DEFAULT_WAVELENGTH,
    cpw: int = 24,
) -> tuple[SolverFIT3D, list[Box], CurrentSource, list]:
    """Build a lightweight 2D periodic unit-cell grating solver (N -> infinity) for schematics."""
    f0 = c0 / wavelength
    freqs = np.array([f0])
    dx_base = wavelength / float(cpw)
    dy_screen = min(dx_base, screen_thickness / 4.0)
    n_x = max(24, 6 * int(round(cpw * (period / wavelength) / 6.0)))
    x = np.linspace(-0.5 * period, 0.5 * period, n_x + 1)

    half_t = 0.5 * screen_thickness
    y_core = graded_axis(
        [-1.4 * wavelength, -half_t, half_t, 1.8 * wavelength],
        [dx_base, dy_screen, dx_base],
        min_cells=[6, 4, 8],
        grading=1.35,
    )
    n_pml = 10
    dy_bot = float(y_core[1] - y_core[0])
    dy_top = float(y_core[-1] - y_core[-2])
    y = np.concatenate(
        [
            y_core[0] - np.arange(n_pml, 0, -1) * dy_bot,
            y_core,
            y_core[-1] + np.arange(1, n_pml + 1) * dy_top,
        ]
    )
    dx_min = float(min(np.min(np.diff(x)), np.min(np.diff(y))))
    lz = 10.0 * dx_min
    z = np.array([-0.5 * lz, 0.5 * lz], dtype=float)

    grid = GridFIT3D(x=x, y=y, z=z, use_mpi=False, verbose=0)
    solver = SolverFIT3D(
        grid,
        bc_low=["periodic", "cpml", "periodic"],
        bc_high=["periodic", "cpml", "periodic"],
        cfln=0.65,
        n_pml=n_pml,
        verbose=0,
    )
    w_bar = 0.5 * (period - slit_width)
    boxes = [
        Box((-0.5 * slit_width - 0.5 * w_bar, 0.0, 0.0), (w_bar, screen_thickness, 2.0 * lz), "pec"),
        Box((0.5 * slit_width + 0.5 * w_bar, 0.0, 0.0), (w_bar, screen_thickness, 2.0 * lz), "pec"),
    ]
    for b in boxes:
        b.material = "pec"
    assign_materials(solver, boxes, f0)

    y_src = float(y_core[0] + 0.35 * wavelength)
    pulse = GaussianSource(frequency=f0, fwidth=0.45 * f0, cutoff=3.5)
    source = CurrentSource(
        src_time=pulse,
        component="x",
        center=(0.0, y_src, 0.0),
        size=(period, 0.0, 2.0 * lz),
        amplitude=1.0,
    )
    y_mon = float(y_core[-3])
    diff_mon = DiffractionMonitor(
        frequencies=freqs,
        center=(0.0, y_mon, 0.0),
        size=(period, 0.0, lz),
    )
    return solver, boxes, source, [diff_mon]


def _fix_unit_spaces_in_texts(ax: plt.Axes) -> None:
    """Ensure a thin space precedes units inside math-mode substrings in schematic text labels."""
    replacements = (
        (" THz$", r"\;\mathrm{THz}$"),
        (" GHz$", r"\;\mathrm{GHz}$"),
        (" MHz$", r"\;\mathrm{MHz}$"),
        (" nm$", r"\;\mathrm{nm}$"),
        (" µm$", r"\;\mu\mathrm{m}$"),
        (" mm$", r"\;\mathrm{mm}$"),
    )
    for txt in ax.texts:
        s = txt.get_text()
        new_s = s
        for old, new in replacements:
            new_s = new_s.replace(old, new)
        if new_s != s:
            txt.set_text(new_s)


def _clean_grating_schematic_labels(
    ax: plt.Axes,
    y_src: float,
    y_mon: float,
    mon_label: str,
    bc_top_summary: str,
) -> None:
    """Clean and reposition labels on a 2D grating setup schematic panel."""
    leg = ax.get_legend()
    if leg is not None:
        leg.remove()
    pec_seen = False
    for txt in ax.texts:
        s = txt.get_text()
        if "$-y$ boundary:" in s or "-y boundary:" in s or r"\pm y" in s:
            txt.set_text(bc_top_summary)
        elif "PEC" in s:
            if not pec_seen:
                txt.set_text(r"PEC grating ($\sigma \to \infty$)")
                txt.set_position((0.0, -0.14))
                txt.set_ha("center")
                txt.set_va("top")
                pec_seen = True
            else:
                txt.set_text("")
        elif "CurrentSource" in s:
            txt.set_position((0.0, y_src - 0.10))
            txt.set_ha("center")
            txt.set_va("top")
        elif "Near2Far" in s or "Diffraction" in s:
            txt.set_text(mon_label)
            txt.set_position((0.0, y_mon + 0.10))
            txt.set_ha("center")
            txt.set_va("bottom")
        elif "FluxMonitor" in s:
            txt.set_text("")
    _fix_unit_spaces_in_texts(ax)


def plot_setup_schematic(
    output_path: str | Path,
    wavelength: float = DEFAULT_WAVELENGTH,
) -> None:
    """Generate the 3-panel labelled setup schematic for finite-N, periodic cell, and mesh zoom."""
    slit_width = 0.6 * wavelength
    period = 1.8 * wavelength
    screen_thickness = 0.14 * wavelength
    a_um = slit_width * 1e6
    d_um = period * 1e6
    b_um = (period - slit_width) * 1e6
    t_um = screen_thickness * 1e6

    solver_fin, grid_info, boxes_fin, src_fin, mons_fin = _build_finite_n_schematic_solver(
        n_slits=6,
        slit_width=slit_width,
        period=period,
        wavelength=wavelength,
        screen_thickness=screen_thickness,
        cpw=22,
    )
    solver_per, boxes_per, src_per, mons_per = _build_periodic_cell_schematic_solver(
        slit_width=slit_width,
        period=period,
        wavelength=wavelength,
        screen_thickness=screen_thickness,
        cpw=24,
    )

    with figstyle.apply_style():
        fig, axes = figstyle.new_figure(nrows=1, ncols=3, width=15.4, height=4.9)

        # Panel (a): Finite N = 6 grating setup in CPML domain
        ax = axes[0]
        schematic.plot_schematic(
            solver_fin,
            ax=ax,
            plane="xy",
            unit="µm",
            structures=boxes_fin,
            sources=[src_fin],
            monitors=mons_fin,
            wavelength=wavelength,
            show_mesh=False,
            mesh_zoom=False,
            show_wave_packet=False,
            title=r"(a) Finite $N = 6$ Grating Setup (CPML + Near2Far)",
        )
        y_src_fin = float(grid_info["y_core"][0] * 1e6 + 0.35 * wavelength * 1e6)
        y_n2f_fin = float(0.5 * t_um + 2.5 * grid_info["dy_screen"] * 1e6)
        _clean_grating_schematic_labels(
            ax,
            y_src=y_src_fin,
            y_mon=y_n2f_fin,
            mon_label="Near2FarMonitor + FluxMonitor",
            bc_top_summary=r"$\pm y$: CPML (10 cells) $\mid$ $\pm z$: Periodic",
        )
        ax.set_aspect("auto")

        # Panel (b): Periodic N -> infinity single unit cell [-d/2, +d/2] with DiffractionMonitor
        ax_p = axes[1]
        schematic.plot_schematic(
            solver_per,
            ax=ax_p,
            plane="xy",
            unit="µm",
            structures=boxes_per,
            sources=[src_per],
            monitors=mons_per,
            wavelength=wavelength,
            show_mesh=False,
            mesh_zoom=False,
            show_wave_packet=False,
            title=r"(b) $N \to \infty$ Unit Cell ($x \in [-d/2, d/2]$ + DiffractionMonitor)",
        )
        _clean_grating_schematic_labels(
            ax_p,
            y_src=-1.05 * wavelength * 1e6,
            y_mon=0.65,
            mon_label="DiffractionMonitor (Floquet orders)",
            bc_top_summary=r"$\pm x$: Periodic $\mid$ $\pm y$: CPML (10 cells)",
        )
        ax_p.set_aspect("auto")

        # Panel (c): Zoomed graded FIT mesh around a grating slit a and PEC bar b = d - a
        ax_m = axes[2]
        ax_m.set_facecolor("#F8FAFC")
        schematic.draw_mesh(
            ax_m,
            solver_fin,
            plane="xy",
            unit="µm",
            every=1,
            wavelength=wavelength,
            zoom_inset=False,
            show_cpw=True,
            color="#334155",
            alpha=0.55,
            linewidth=0.6,
        )
        schematic.overlay_geometry(
            ax_m,
            solver_fin,
            plane="xy",
            unit="µm",
            structures=boxes_fin,
            show_boundaries=False,
            show_materials=True,
            fill_materials=True,
            material_alpha=0.55,
            outline_color=figstyle.INK,
            label_materials=False,
        )
        ax_m.set_xlim(-1.15, 1.15)
        ax_m.set_ylim(-0.32, 0.32)
        ax_m.set_aspect("auto")
        ax_m.set_xlabel("x [µm]")
        ax_m.set_ylabel("y [µm]")
        ax_m.set_title(r"(c) Grating Mesh Zoom ($a = 0.6\lambda$, $b = 1.2\lambda$, $d = 1.8\lambda$)")

        ax_m.text(
            0.0,
            0.0,
            f"PEC bar $b = {b_um:.2f}$ µm\n({grid_info['cells_across_bridge']} cells $\\geq 4$)",
            ha="center",
            va="center",
            fontsize=7.8,
            fontweight="bold",
            color=figstyle.INK,
            bbox={
                "boxstyle": "round,pad=0.15",
                "facecolor": figstyle.PAPER,
                "edgecolor": "#94A3B8",
                "alpha": 0.92,
            },
            zorder=9,
        )
        for x_c in (-0.5 * d_um, 0.5 * d_um):
            ax_m.annotate(
                "",
                xy=(x_c + 0.5 * a_um, 0.14),
                xytext=(x_c - 0.5 * a_um, 0.14),
                arrowprops={"arrowstyle": "<->", "color": "#0072B2", "lw": 1.3},
                zorder=9,
            )
            ax_m.text(
                x_c,
                0.16,
                f"Slit $a = {a_um:.2f}$ µm\n({grid_info['cells_across_slit']} cells $\\geq 4$)",
                ha="center",
                va="bottom",
                fontsize=7.8,
                fontweight="bold",
                color="#0072B2",
                zorder=9,
            )
        ax_m.text(
            -0.95,
            -0.14,
            f"Screen $t = {t_um*1e3:.0f}$ nm ({grid_info['cells_across_screen']} cells $\\geq 4$)",
            ha="left",
            va="top",
            fontsize=8.0,
            fontweight="bold",
            color="#D55E00",
            zorder=9,
        )
        _fix_unit_spaces_in_texts(ax_m)

        figstyle.save(fig, output_path, close=True)


def plot_n_slit_patterns_and_nearfield(
    pat_data: dict,
    output_path: str | Path,
) -> None:
    """Generate the 3-panel figure for N-slit near field, far-field patterns, and (N-2) secondary maxima."""
    wavelength = DEFAULT_WAVELENGTH
    lam_um = wavelength * 1e6
    solver_fin, _, boxes_fin, _, _ = _build_finite_n_schematic_solver(
        n_slits=6,
        slit_width=0.6 * wavelength,
        period=1.8 * wavelength,
        wavelength=wavelength,
        screen_thickness=0.14 * wavelength,
        cpw=22,
    )

    with figstyle.apply_style():
        fig, axes = figstyle.new_figure(nrows=1, ncols=3, width=15.2, height=4.8)

        # Panel (a): 2D near-field intensity map |H_z(x, y)|^2 / |H_inc|^2 for N = 6
        ax = axes[0]
        x_um = pat_data["map_x_over_lam"] * lam_um
        y_um = pat_data["map_y_over_lam"] * lam_um
        inten = pat_data["map_intensity"]
        im = ax.pcolormesh(
            x_um,
            y_um,
            inten.T,
            shading="auto",
            cmap=figstyle.CMAP_INTENSITY,
            vmin=0.0,
            vmax=min(float(np.percentile(inten, 99.5)), 3.2),
        )
        fig.colorbar(im, ax=ax, label=r"$|H_z(x, y)|^2 / |H_{\mathrm{inc}}|^2$ [-]")
        schematic.overlay_geometry(
            ax,
            solver_fin,
            plane="xy",
            unit="µm",
            orientation="2d",
            structures=boxes_fin,
            show_boundaries=True,
            show_materials=True,
            fill_materials=True,
            material_alpha=0.38,
            outline_color="#FFFFFF",
            label_materials=False,
        )
        ax.text(
            0.0,
            0.12,
            "PEC grating ($N = 6$)",
            color="white",
            fontsize=7.6,
            fontweight="bold",
            ha="center",
            va="bottom",
            zorder=8,
        )
        ax.set_xlim(float(x_um[0]), float(x_um[-1]))
        ax.set_ylim(float(y_um[0]), float(y_um[-1]))
        ax.set_aspect("auto")
        ax.set_xlabel(r"Transverse coordinate $x$ [µm]")
        ax.set_ylabel(r"Propagation coordinate $y$ [µm]")
        ax.set_title(r"(a) 2D FIT Talbot carpet ($N = 6$, $a = 0.6\lambda$, $d = 1.8\lambda$)")

        # Panel (b): Far-field patterns for N = 2, 4, 8 showing principal peaks & envelope
        ax = axes[1]
        theta_deg = pat_data["theta_deg"]
        n_vals = list(pat_data["n_values"])
        show_indices = list(dict.fromkeys([0, min(2, len(n_vals) - 1), len(n_vals) - 1]))
        for idx_plot, k in enumerate(show_indices):
            n_val = int(n_vals[k])
            off = idx_plot * 1.12
            err = float(pat_data["l2_errors"][k]) * 100.0
            ax.plot(
                theta_deg,
                pat_data["patterns_fit"][k] + off,
                color=figstyle.PALETTE[idx_plot % len(figstyle.PALETTE)],
                lw=2.0,
                ls="-",
                label=rf"FIT $N = {n_val}$ ($L^2={err:.1f}\%$)",
            )
            ax.plot(
                theta_deg,
                pat_data["patterns_theory"][k] + off,
                color=figstyle.INK,
                lw=1.1,
                ls="--",
                label=r"Theory $[\sin(N\alpha)/(N\sin\alpha)]^2\mathrm{sinc}^2\beta$"
                if idx_plot == 0
                else "_nolegend_",
            )
            ax.plot(
                theta_deg,
                pat_data["single_env_theory"] + off,
                color="#D55E00",
                lw=1.2,
                ls=":",
                label=r"Envelope $\mathrm{sinc}^2\beta$" if idx_plot == 0 else "_nolegend_",
            )
        theta_m1 = float(pat_data["theta_m1_theory"][0])
        ax.axvline(theta_m1, color="#CC79A7", ls="-.", lw=1.1)
        ax.axvline(
            -theta_m1,
            color="#CC79A7",
            ls="-.",
            lw=1.1,
            label=rf"Orders $m=\pm 1$ ($\pm {theta_m1:.1f}^\circ$)",
        )
        ax.set_xlim(-76.0, 76.0)
        ax.set_ylim(-0.05, 4.05)
        ax.set_xlabel(r"Diffraction angle $\theta$ [deg]")
        ax.set_ylabel(
            r"Normalized far-field intensity $U_{\mathrm{2D}}(\theta)/U_{\mathrm{2D}}(0)$ (offset) [-]"
        )
        ax.set_title(r"(b) Sharpening of grating orders with $N$ ($d/a = 3$)")
        figstyle.spectrum_grid(ax)
        ax.legend(loc="upper center", ncol=2, fontsize=6.6)

        # Panel (c): Zoom between m = 0 and m = +1 showing (N - 2) secondary maxima & peak suppression
        ax = axes[2]
        sin_norm = np.sin(np.deg2rad(theta_deg)) * float(pat_data["period_over_lam"])
        zoom_mask = (sin_norm >= -0.15) & (sin_norm <= 1.15)
        for k, n_val in enumerate(n_vals):
            off = k * 0.24
            n_sec = int(pat_data["n_sec_counted"][k])
            max_sec_pct = float(pat_data["max_sec_fit"][k]) * 100.0
            sec_lbl = (
                rf"FIT $N={int(n_val)}$ ($N-2={n_sec}$)"
                if int(n_val) == 2
                else rf"FIT $N={int(n_val)}$ ($N-2={n_sec}$, {max_sec_pct:.1f}%)"
            )
            ax.plot(
                sin_norm[zoom_mask],
                pat_data["patterns_fit"][k][zoom_mask] + off,
                color=figstyle.PALETTE[k % len(figstyle.PALETTE)],
                lw=2.0,
                ls="-",
                label=sec_lbl,
            )
            ax.plot(
                sin_norm[zoom_mask],
                pat_data["patterns_theory"][k][zoom_mask] + off,
                color=figstyle.INK,
                lw=1.1,
                ls="--",
                label="Grating theory" if k == 0 else "_nolegend_",
            )
        ax.axvline(0.0, color="#64748B", ls=":", lw=1.1)
        ax.axvline(1.0, color="#64748B", ls=":", lw=1.1)
        ax.set_xlim(-0.12, 1.12)
        ax.set_ylim(-0.04, 1.35 + 0.24 * (len(n_vals) - 1))
        ax.set_xlabel(r"Normalized direction cosine $(d / \lambda)\sin\theta$ (order $m$) [-]")
        ax.set_ylabel(
            r"Normalized intensity $U_{\mathrm{2D}}(\theta)/U_{\mathrm{2D}}(0)$ (offset) [-]"
        )
        ax.set_title(r"(c) $(N - 2)$ secondary maxima ($m = 0 \to +1$)")
        figstyle.spectrum_grid(ax)
        ax.legend(loc="upper center", ncol=2, fontsize=6.6)

        figstyle.save(fig, output_path, close=True)


def plot_resolving_power_and_width(
    pat_data: dict,
    res_pow_data: dict,
    output_path: str | Path,
) -> None:
    """Generate the 2-panel figure for 1/N peak width scaling and Rayleigh chromatic resolving power."""
    with figstyle.apply_style():
        fig, axes = figstyle.new_figure(nrows=1, ncols=2, width=13.8, height=4.8)

        # Panel (a): Principal peak width Delta(sin theta) ~ 1/N (both HWZM = lambda/(N d) and FWHM)
        ax = axes[0]
        n_vals = pat_data["n_values"]
        d_over_lam = float(pat_data["period_over_lam"])
        n_fine = np.linspace(1.8, max(float(n_vals[-1]) + 0.5, 8.5), 161)
        ax.plot(
            n_fine,
            1.0 / (n_fine * d_over_lam),
            color=figstyle.INK,
            lw=1.8,
            ls="--",
            label=r"Exact first-zero width $\Delta(\sin\theta)_{\mathrm{HWZM}} = \lambda / (N d)$",
        )
        ax.plot(
            n_fine,
            0.88589 / (n_fine * d_over_lam),
            color=figstyle.PALETTE[1],
            lw=1.6,
            ls=":",
            label=r"Asymptotic FWHM $\Delta(\sin\theta)_{\mathrm{FWHM}} \approx 0.886\,\lambda / (N d)$",
        )
        ax.plot(
            n_vals,
            pat_data["hwzm_sin_fit"],
            "o",
            color=figstyle.PALETTE[0],
            ms=6.5,
            label=r"2D FIT measured $\Delta(\sin\theta)_{\mathrm{HWZM}}$",
        )
        ax.plot(
            n_vals,
            pat_data["fwhm_sin_fit"],
            "s",
            color=figstyle.PALETTE[1],
            ms=6.0,
            label=r"2D FIT measured $\Delta(\sin\theta)_{\mathrm{FWHM}}$",
        )
        ax.set_xlabel(r"Number of slits $N$ [-]")
        ax.set_ylabel(r"Principal maximum angular width $\Delta(\sin\theta)$ [-]")
        ax.set_title(
            r"(a) Principal peak narrowing $\Delta(\sin\theta) \propto 1 / N$ ($d = 1.8\lambda$)"
        )
        figstyle.spectrum_grid(ax)
        ax.legend(loc="upper right", fontsize=7.8)

        # Panel (b): Rayleigh chromatic resolving power R = lambda / Delta_lambda = |m| N (N=6, m=+1)
        ax = axes[1]
        theta_deg = res_pow_data["theta_deg"]
        saddles = res_pow_data["saddle_to_peak_fit"]
        labels = [
            rf"Unresolved $\Delta\lambda/\lambda = 0.55/R$ (saddle/peak $={saddles[0]:.2f}$)",
            rf"Rayleigh limit $\Delta\lambda/\lambda = 1/R = 1/6$ (saddle/peak $={saddles[1]:.2f}$)",
            rf"Well-resolved $\Delta\lambda/\lambda = 1.45/R$ (saddle/peak $={saddles[2]:.2f}$)",
        ]
        for k in range(3):
            ax.plot(
                theta_deg,
                res_pow_data["doublet_sum_fit"][k],
                color=figstyle.PALETTE[k],
                lw=2.0,
                ls="-",
                label="FIT " + labels[k],
            )
            ax.plot(
                theta_deg,
                res_pow_data["doublet_sum_th"][k],
                color=figstyle.INK,
                lw=1.1,
                ls="--",
                label="Theory doublet sum" if k == 0 else "_nolegend_",
            )
        ax.plot(
            theta_deg,
            res_pow_data["u_base_fit"],
            color=figstyle.PALETTE[0],
            lw=1.2,
            ls=":",
            alpha=0.85,
            label=r"Component $\lambda_0$ ($m=+1$)",
        )
        ax.plot(
            theta_deg,
            res_pow_data["u_shifted_fit"][1],
            color=figstyle.PALETTE[4],
            lw=1.2,
            ls=":",
            alpha=0.85,
            label=r"Component $\lambda_0(1 + 1/6)$ ($m=+1$)",
        )
        ax.set_xlim(24.0, 50.0)
        ax.set_ylim(-0.04, 1.85)
        ax.set_xlabel(r"Diffraction angle $\theta$ around order $m = +1$ [deg]")
        ax.set_ylabel(r"Doublet intensity $I(\theta, \lambda_1) + I(\theta, \lambda_2)$ [arb.]")
        ax.set_title(
            r"(b) Rayleigh resolving power $R = \lambda/\Delta\lambda = |m|N = 6$ ($N=6, m=+1$)"
        )
        figstyle.spectrum_grid(ax)
        ax.legend(loc="upper right", fontsize=6.9)

        figstyle.save(fig, output_path, close=True)


def plot_periodic_comparison_and_cost(
    pat_data: dict,
    per_data: dict,
    res_data: dict,
    output_path: str | Path,
) -> None:
    """Generate the 3-panel figure for finite-N vs periodic Bloch unit cell, cost scaling, and convergence."""
    with figstyle.apply_style():
        fig, axes = figstyle.new_figure(nrows=1, ncols=3, width=15.2, height=4.8)

        # Panel (a): Finite-N integrated lobe fractions vs N -> infinity periodic unit-cell DiffractionMonitor
        ax = axes[0]
        n_vals = pat_data["n_values"]
        lobe_fracs = pat_data["lobe_fractions"]  # shape (len(n_vals), 3) for m = -1, 0, +1
        per_fracs = per_data["periodic_order_fractions"]  # shape (3,) for m = -1, 0, +1
        ax.plot(
            n_vals,
            lobe_fracs[:, 1],
            "o-",
            color=figstyle.PALETTE[0],
            lw=2.0,
            ms=6.0,
            label=r"Finite-$N$ integrated lobe $m = 0$",
        )
        ax.plot(
            n_vals,
            0.5 * (lobe_fracs[:, 0] + lobe_fracs[:, 2]),
            "s-",
            color=figstyle.PALETTE[1],
            lw=2.0,
            ms=6.0,
            label=r"Finite-$N$ integrated lobe $m = \pm 1$",
        )
        ax.axhline(
            float(per_fracs[1]),
            color=figstyle.PALETTE[0],
            ls="--",
            lw=1.6,
            label=rf"$N \to \infty$ periodic cell $T_0 / T_{{\mathrm{{tot}}}} = {float(per_fracs[1]):.3f}$",
        )
        ax.axhline(
            float(0.5 * (per_fracs[0] + per_fracs[2])),
            color=figstyle.PALETTE[1],
            ls="--",
            lw=1.6,
            label=rf"$N \to \infty$ periodic cell $T_{{\pm 1}} / T_{{\mathrm{{tot}}}} = {float(0.5*(per_fracs[0]+per_fracs[2])):.3f}$",
        )
        ax.set_xlabel(r"Number of slits $N$ [-]")
        ax.set_ylabel(r"Normalized order power fraction $\eta_m / \sum_k \eta_k$ [-]")
        ax.set_title(r"(a) Finite-$N$ vs $N \to \infty$ periodic unit cell")
        figstyle.spectrum_grid(ax)
        ax.legend(loc="center right", fontsize=7.5)

        # Panel (b): Computational cost scaling (2D cells & wall-clock / RAM) of finite N vs 1 periodic cell
        ax = axes[1]
        ax.plot(
            per_data["n_proj"],
            per_data["proj_cells"] / 1e3,
            "o--",
            color=figstyle.PALETTE[3],
            lw=1.8,
            ms=5.5,
            label=r"Finite-$N$ domain (wakis.estimate cells)",
        )
        ax.plot(
            pat_data["n_values"],
            pat_data["n_cells"] / 1e3,
            "s",
            color=figstyle.PALETTE[0],
            ms=6.5,
            label=r"Finite-$N$ executed 2D FIT runs",
        )
        ax.axhline(
            float(per_data["periodic_n_cells"]) / 1e3,
            color=figstyle.PALETTE[2],
            ls="-.",
            lw=2.0,
            label=rf"1 periodic cell ($N\to\infty$, Ch. 6): {float(per_data['periodic_n_cells'])/1e3:.1f}k cells",
        )
        ax.set_xlabel(r"Number of slits $N$ in finite domain [-]")
        ax.set_ylabel(r"2D mesh size $N_{\mathrm{cells}} / 10^3$ [kcells]")
        ax.set_title(r"(b) Cost of finite $N$ vs periodic unit cell (Ch. 6)")
        figstyle.spectrum_grid(ax)
        ax.legend(loc="upper left", fontsize=7.5)

        # Panel (c): Mesh resolution convergence study (N = 4)
        ax = axes[2]
        cpw = res_data["cpw_values"]
        ax.plot(
            cpw,
            res_data["l2_vs_theory"] * 100.0,
            "o-",
            color=figstyle.PALETTE[0],
            lw=2.0,
            ms=5.5,
            label=r"Far-field $L^2$ error vs $N$-slit theory [%]",
        )
        ax.plot(
            cpw,
            res_data["rel_to_finest"] * 100.0,
            "s--",
            color=figstyle.PALETTE[2],
            lw=1.8,
            ms=5.5,
            label=r"Relative $L^2$ diff vs finest mesh [%]",
        )
        ax.axvline(
            20.0,
            color="#64748B",
            ls=":",
            lw=1.2,
            label=r"Chapter 3 rule: $\mathrm{cpw} \geq 20$",
        )
        ax.set_xlabel(r"Background resolution $\mathrm{cpw} = \lambda / \Delta x$ [cells/$\lambda$]")
        ax.set_ylabel("Relative error / discrepancy [%]")
        ax.set_title(r"(c) Mesh convergence ($N = 4$, $a = 0.6\lambda$, $d = 1.8\lambda$)")
        figstyle.spectrum_grid(ax)
        ax.legend(loc="center right", fontsize=7.8)

        figstyle.save(fig, output_path, close=True)
