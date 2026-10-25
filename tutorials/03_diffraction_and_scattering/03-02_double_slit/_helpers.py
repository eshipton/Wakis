"""Helper functions for Tutorial 3.2: Young's Double-Slit Interference and Slit Coupling."""

from __future__ import annotations

from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c as c0
from scipy.special import hankel1

from wakis import DFTMonitor, FluxMonitor, GridFIT3D, SolverFIT3D, figstyle, schematic
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


def single_slit_envelope(
    theta: np.ndarray,
    slit_width: float,
    wavelength: float = DEFAULT_WAVELENGTH,
    obliquity: str = "rs1",
) -> np.ndarray:
    """Single-slit far-field diffraction envelope I_env(theta) / I_env(0).

    Parameters
    ----------
    theta : ndarray
        Diffraction angle measured from the +y normal [rad].
    slit_width : float
        Individual slit opening width a [m].
    wavelength : float, optional
        Free-space wavelength lambda [m].
    obliquity : {"none", "rs1", "rs2", "kirchhoff"}, optional
        Obliquity factor: ``"rs1"`` for cos^2(theta) (E || slit edges), ``"rs2"`` or
        ``"none"`` for 1.0 (Fraunhofer / E perp slit edges), ``"kirchhoff"`` for
        ((1 + cos(theta))/2)^2.

    Returns
    -------
    ndarray
        Normalized single-slit envelope in [0, 1] [-].
    """
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


def interference_factor(
    theta: np.ndarray,
    slit_sep: float,
    wavelength: float = DEFAULT_WAVELENGTH,
) -> np.ndarray:
    """Two-slit interference factor cos^2(alpha) with alpha = pi * d * sin(theta) / lambda.

    Parameters
    ----------
    theta : ndarray
        Diffraction angle measured from the +y normal [rad].
    slit_sep : float
        Center-to-center slit separation d [m].
    wavelength : float, optional
        Free-space wavelength lambda [m].

    Returns
    -------
    ndarray
        Normalized two-slit interference factor in [0, 1] [-].
    """
    theta_arr = np.asarray(theta, dtype=float)
    alpha = np.pi * (slit_sep / wavelength) * np.sin(theta_arr)
    return np.cos(alpha) ** 2


def double_slit_farfield_pattern(
    theta: np.ndarray,
    slit_width: float,
    slit_sep: float,
    wavelength: float = DEFAULT_WAVELENGTH,
    obliquity: str = "rs1",
) -> np.ndarray:
    """Normalized double-slit far-field intensity cos^2(alpha) * sinc^2(beta) * Omega(theta).

    Parameters
    ----------
    theta : ndarray
        Diffraction angle measured from the +y normal [rad].
    slit_width : float
        Individual slit opening width a [m].
    slit_sep : float
        Center-to-center slit separation d [m].
    wavelength : float, optional
        Free-space wavelength lambda [m].
    obliquity : {"none", "rs1", "rs2", "kirchhoff"}, optional
        Obliquity factor convention.

    Returns
    -------
    ndarray
        Normalized angular intensity I(theta) / I(0) [-].
    """
    return interference_factor(theta, slit_sep, wavelength) * single_slit_envelope(
        theta, slit_width, wavelength, obliquity=obliquity
    )


def fringe_spacing_sin_theta(
    slit_sep: float,
    wavelength: float = DEFAULT_WAVELENGTH,
) -> float:
    """Return the exact angular fringe spacing in sin(theta) space: Delta(sin(theta)) = lambda / d [-]."""
    if slit_sep <= 0.0 or wavelength <= 0.0:
        raise ValueError("slit_sep > 0 and wavelength > 0 are required.")
    return float(wavelength / slit_sep)


def missing_order_indices(
    slit_width: float,
    slit_sep: float,
    max_order: int = 12,
    tol: float = 1e-6,
) -> list[int]:
    """Return positive interference orders m > 0 extinguished by a single-slit diffraction null.

    An interference maximum at sin(theta_m) = m * lambda / d coincides with a
    single-slit diffraction zero at sin(theta_n) = n * lambda / a whenever
    m * (a / d) = n is a non-zero integer.

    Parameters
    ----------
    slit_width : float
        Slit opening width a [m].
    slit_sep : float
        Center-to-center slit separation d [m].
    max_order : int, optional
        Largest positive interference order m to check [-].
    tol : float, optional
        Tolerance on integer ratio matching [-].

    Returns
    -------
    list of int
        Positive missing interference order indices m [-].
    """
    if slit_width <= 0.0 or slit_sep <= slit_width:
        raise ValueError("Require 0 < slit_width < slit_sep.")
    out: list[int] = []
    for m in range(1, int(max_order) + 1):
        n_float = m * (slit_width / slit_sep)
        n_int = int(round(n_float))
        if n_int >= 1 and abs(n_float - n_int) <= tol:
            out.append(m)
    return out


def double_slit_rayleigh_sommerfeld_line(
    x: np.ndarray,
    slit_width: float,
    slit_sep: float,
    wavelength: float,
    distance: float,
    kind: str = "I",
    n_quad_per_slit: int = 201,
) -> np.ndarray:
    """Exact 2D Rayleigh-Sommerfeld intensity profile on a line at distance L downstream.

    Integrates the 2D Hankel Green's function over the two slits
    [-d/2 - a/2, -d/2 + a/2] and [+d/2 - a/2, +d/2 + a/2].

    Parameters
    ----------
    x : ndarray
        Transverse detector coordinates [m].
    slit_width : float
        Individual slit width a [m].
    slit_sep : float
        Center-to-center slit separation d [m].
    wavelength : float
        Free-space wavelength lambda [m].
    distance : float
        Downstream propagation distance L from the screen exit face [m].
    kind : {"I", "II"}, optional
        "I" for Dirichlet (E_z) or "II" for Neumann (H_z).
    n_quad_per_slit : int, optional
        Gauss-Legendre quadrature nodes per slit [-].

    Returns
    -------
    ndarray
        Intensity |psi(x, L)|^2 normalized by the incident wave intensity [-].
    """
    x_arr = np.asarray(x, dtype=float)
    k0 = 2.0 * np.pi / wavelength
    xi, wi = np.polynomial.legendre.leggauss(int(n_quad_per_slit))
    x_local = 0.5 * slit_width * xi
    w_local = 0.5 * slit_width * wi

    x_prime = np.concatenate([x_local - 0.5 * slit_sep, x_local + 0.5 * slit_sep])
    weights = np.concatenate([w_local, w_local])

    dx = x_arr[:, None] - x_prime[None, :]
    r = np.hypot(dx, distance)
    kr = k0 * r

    if kind.upper() == "I":
        kernel = (k0 * distance / (2.0j)) * (hankel1(1, kr) / r)
    elif kind.upper() == "II":
        kernel = 0.5 * k0 * hankel1(0, kr)
    else:
        raise ValueError(f"Unknown Rayleigh-Sommerfeld kind {kind!r}.")

    field = kernel @ weights
    return np.abs(field) ** 2


def surface_wave_coupling_model(
    d_over_lam: np.ndarray,
    amp_hz: float = 0.19,
    phase_hz: float = 0.58 * np.pi,
    amp_ez: float = 0.04,
    phase_ez: float = 0.15 * np.pi,
) -> dict[str, np.ndarray]:
    """Asymptotic cylindrical surface-wave model for two-slit transmission modulation.

    For E perp slit edges (H_z, p-polarization), each slit diffracts grazing magnetic
    fields along the PEC surface with 2D Hankel amplitude ~ H_0^(1)(k_0 d) ~
    exp(i k_0 d - i pi/4) / sqrt(k_0 d), modulating the total transmitted power
    around 2 * P'_single with period Delta d ~ lambda (Schouten et al., PRL 94, 053901,
    2005). For E || slit edges (E_z, s-polarization), the Dirichlet condition E_z = 0
    on the PEC jaw forces the surface field to decay as (k_0 d)^(-3/2), strongly
    suppressing inter-slit coupling.
    """
    u = np.asarray(d_over_lam, dtype=float)
    eta_hz = 1.0 + amp_hz * np.cos(2.0 * np.pi * u + phase_hz) / np.sqrt(np.maximum(u, 0.2))
    eta_ez = 1.0 + amp_ez * np.cos(2.0 * np.pi * u + phase_ez) / (np.maximum(u, 0.2) ** 1.5)
    return {"eta_hz_model": eta_hz, "eta_ez_model": eta_ez}


def build_double_slit_grid(
    slit_width: float,
    slit_sep: float,
    wavelength: float = DEFAULT_WAVELENGTH,
    screen_thickness: float = 0.15 * DEFAULT_WAVELENGTH,
    x_half_span: float | None = None,
    y_upstream: float = 1.3 * DEFAULT_WAVELENGTH,
    y_downstream: float = 4.2 * DEFAULT_WAVELENGTH,
    cpw: int = 22,
    min_slit_cells: int = 6,
    min_barrier_cells: int = 4,
    min_screen_cells: int = 4,
    n_pml: int = 10,
) -> dict:
    """Build a symmetric 2D graded FIT mesh snapped to both slits and the central barrier.

    Enforces Chapter 3 discretization rules:
    - ``cpw >= 20`` in standard runs,
    - ``>= min_slit_cells`` (>= 4) across each slit opening,
    - ``>= min_barrier_cells`` (>= 4) across the central PEC bridge width ``b = d - a``,
    - ``>= min_screen_cells`` (>= 4) across the PEC screen thickness ``t``.
    """
    bridge_width = slit_sep - slit_width
    if bridge_width <= 0.0:
        raise ValueError(f"Require slit_sep > slit_width; got d={slit_sep}, a={slit_width}.")

    x_outer = 0.5 * slit_sep + 0.5 * slit_width
    x_inner = 0.5 * slit_sep - 0.5 * slit_width
    if x_half_span is None:
        x_half_span = max(5.5 * wavelength, x_outer + 3.0 * wavelength)

    dx_base = wavelength / float(cpw)
    dx_slit = min(dx_base, slit_width / float(max(min_slit_cells, 4)))
    dx_bridge = min(dx_base, bridge_width / float(max(min_barrier_cells, 4)))
    dy_screen = min(dx_base, screen_thickness / float(max(min_screen_cells, 4)))

    x_pts = [
        -x_half_span,
        -x_outer,
        -x_inner,
        x_inner,
        x_outer,
        x_half_span,
    ]
    x_sizes = [dx_base, dx_slit, dx_bridge, dx_slit, dx_base]
    x_min_c = [
        4,
        max(min_slit_cells, 4),
        max(min_barrier_cells, 4),
        max(min_slit_cells, 4),
        4,
    ]
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

    right_slit_mask = (x_core >= x_inner - 1e-15) & (x_core <= x_outer + 1e-15)
    bridge_mask = (x_core >= -x_inner - 1e-15) & (x_core <= x_inner + 1e-15)
    scr_mask = (y_core >= -half_t - 1e-15) & (y_core <= half_t + 1e-15)

    return {
        "x": x,
        "y": y,
        "z": z,
        "x_core": x_core,
        "y_core": y_core,
        "lz": lz,
        "dx_base": dx_base,
        "dx_slit": dx_slit,
        "dy_screen": dy_screen,
        "n_pml": int(n_pml),
        "cells_across_slit": int(max(np.count_nonzero(right_slit_mask) - 1, 0)),
        "cells_across_bridge": int(max(np.count_nonzero(bridge_mask) - 1, 0)),
        "cells_across_screen": int(max(np.count_nonzero(scr_mask) - 1, 0)),
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


def run_double_slit_simulation(
    slit_width: float,
    slit_sep: float,
    wavelength: float = DEFAULT_WAVELENGTH,
    screen_thickness: float = 0.15 * DEFAULT_WAVELENGTH,
    polarization: str = "Ez",
    geometry_mode: str = "double",
    cpw: int = 22,
    min_slit_cells: int = 6,
    min_barrier_cells: int = 4,
    min_screen_cells: int = 4,
    x_half_span: float | None = None,
    y_upstream: float = 1.3 * DEFAULT_WAVELENGTH,
    y_downstream: float = 4.2 * DEFAULT_WAVELENGTH,
    detector_distance: float | None = 3.2 * DEFAULT_WAVELENGTH,
    record_2d_dft: bool = False,
    theta_deg: np.ndarray | None = None,
    n_cycles: float = 13.0,
) -> dict:
    """Run a 2D FIT double-slit, single-slit reference, or empty-domain simulation."""
    f0 = c0 / wavelength
    grid_info = build_double_slit_grid(
        slit_width=slit_width,
        slit_sep=slit_sep,
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
    n_pml = int(grid_info["n_pml"])
    half_t = 0.5 * screen_thickness

    pol = polarization.strip()
    if pol not in ("Ez", "Hz"):
        raise ValueError(f"polarization must be 'Ez' or 'Hz', got {polarization!r}.")
    src_component = "z" if pol == "Ez" else "x"
    dft_component = "Ez" if pol == "Ez" else "Hz"

    left_end = float(x[0] - 2.0 * wavelength)
    right_end = float(x[-1] + 2.0 * wavelength)
    boxes = []
    if geometry_mode == "double":
        x_outer = 0.5 * slit_sep + 0.5 * slit_width
        x_inner = 0.5 * slit_sep - 0.5 * slit_width
        w_left = (-x_outer) - left_end
        xc_left = 0.5 * (left_end - x_outer)
        w_right = right_end - x_outer
        xc_right = 0.5 * (x_outer + right_end)
        w_bridge = 2.0 * x_inner
        boxes = [
            Box((xc_left, 0.0, 0.0), (w_left, screen_thickness, 2.0 * lz), "pec"),
            Box((0.0, 0.0, 0.0), (w_bridge, screen_thickness, 2.0 * lz), "pec"),
            Box((xc_right, 0.0, 0.0), (w_right, screen_thickness, 2.0 * lz), "pec"),
        ]
    elif geometry_mode == "single_centered":
        w_left = (-0.5 * slit_width) - left_end
        xc_left = 0.5 * (left_end - 0.5 * slit_width)
        w_right = right_end - 0.5 * slit_width
        xc_right = 0.5 * (0.5 * slit_width + right_end)
        boxes = [
            Box((xc_left, 0.0, 0.0), (w_left, screen_thickness, 2.0 * lz), "pec"),
            Box((xc_right, 0.0, 0.0), (w_right, screen_thickness, 2.0 * lz), "pec"),
        ]
    elif geometry_mode == "single_right":
        x_outer = 0.5 * slit_sep + 0.5 * slit_width
        x_inner = 0.5 * slit_sep - 0.5 * slit_width
        w_left = x_inner - left_end
        xc_left = 0.5 * (left_end + x_inner)
        w_right = right_end - x_outer
        xc_right = 0.5 * (x_outer + right_end)
        boxes = [
            Box((xc_left, 0.0, 0.0), (w_left, screen_thickness, 2.0 * lz), "pec"),
            Box((xc_right, 0.0, 0.0), (w_right, screen_thickness, 2.0 * lz), "pec"),
        ]
    elif geometry_mode != "empty":
        raise ValueError(f"Unknown geometry_mode {geometry_mode!r}.")

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
    pulse = GaussianSource(frequency=f0, fwidth=0.5 * f0, cutoff=3.5)
    source = CurrentSource(
        src_time=pulse,
        component=src_component,
        center=(0.0, y_src, 0.0),
        size=(src_span_x, 0.0, 2.0 * lz),
        amplitude=1.0,
    )

    freqs = np.array([f0])
    w_ref = float((x_core[-1] - x_core[0]) - 0.5 * wavelength)
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

    mon_det = None
    if detector_distance is not None:
        y_det = float(half_t + detector_distance)
        mon_det = DFTMonitor(
            frequencies=freqs,
            components=(dft_component,),
            center=(0.0, y_det, 0.0),
            size=(w_ref, 0.0, 0.0),
        )
        monitors.append(mon_det)

    dft_2d = None
    if record_2d_dft:
        y_span_core = float(y_core[-1] - y_core[0])
        y_mid_core = 0.5 * float(y_core[-1] + y_core[0])
        dft_2d = DFTMonitor(
            frequencies=freqs,
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
        theta_deg = np.linspace(-82.0, 82.0, 657)
    theta_rad = np.deg2rad(theta_deg)

    s_f0 = complex(pulse.fourier_transform(f0))
    power_norm = float(2.0 * (abs(s_f0) ** 2))

    u2d = _compute_2d_farfield_from_n2f(n2f, theta_rad, wavelength, lz, ifreq=0) / power_norm
    u2d_integral = float(np.trapezoid(u2d, theta_rad))
    p_wide_per_z = float(flux_wide.get_flux()[0] / (lz * power_norm))
    p_n2f_per_z = float(n2f.get_flux()[0] / (lz * power_norm))

    det_profile = None
    if mon_det is not None:
        fld_det, coords_det = mon_det.get_field(dft_component)
        det_profile = {
            "x": coords_det[0].copy(),
            "field": (np.squeeze(fld_det[0]) / s_f0).copy(),
        }

    map_2d = None
    if dft_2d is not None:
        fld_2d, coords_2d = dft_2d.get_field(dft_component)
        map_2d = {
            "x": coords_2d[0].copy(),
            "y": coords_2d[1].copy(),
            "field": (np.squeeze(fld_2d[0]) / s_f0).copy(),
        }

    return {
        "slit_width": float(slit_width),
        "slit_sep": float(slit_sep),
        "wavelength": float(wavelength),
        "screen_thickness": float(screen_thickness),
        "polarization": pol,
        "geometry_mode": geometry_mode,
        "cpw": int(cpw),
        "n_cells": int(grid.Nx * grid.Ny * grid.Nz),
        "n_steps": int(n_steps),
        "dt": float(solver.dt),
        "elapsed_s": float(elapsed_s),
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
        "det_profile": det_profile,
        "map_2d": map_2d,
    }


def run_preflight_summary(
    wavelength: float = DEFAULT_WAVELENGTH,
    slit_width: float = 1.5 * DEFAULT_WAVELENGTH,
    slit_sep: float = 4.5 * DEFAULT_WAVELENGTH,
    screen_thickness: float = 0.15 * DEFAULT_WAVELENGTH,
    cpw: int = 22,
) -> dict:
    """Run ``wakis.estimate.estimate`` and ``preflight`` on the representative double-slit mesh."""
    grid_info = build_double_slit_grid(
        slit_width=slit_width,
        slit_sep=slit_sep,
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


def run_missing_orders_task(quick: bool = False) -> dict:
    """Compare cos^2 * sinc^2 envelope and missing orders for d / a in {2, 3, 4}.

    With slit width a = 1.5 * lambda, the first single-slit diffraction null lies at
    sin(theta_null) = lambda / a = 2/3 (theta_null = +/- 41.81 deg). Consequently, for
    d / a = p in {2, 3, 4}, the p-th interference order at sin(theta_p) = p * lambda / d
    = lambda / a coincides with the diffraction zero and is extinguished.
    """
    wavelength = DEFAULT_WAVELENGTH
    slit_width = 1.5 * wavelength
    screen_thickness = 0.14 * wavelength
    cpw = 20 if quick else 22
    p_ratios = [2, 3, 4]
    det_dist = 3.5 * wavelength
    x_span = 6.2 * wavelength

    theta_deg = np.linspace(-80.0, 80.0, 641)
    theta_rad = np.deg2rad(theta_deg)
    idx0 = int(np.argmin(np.abs(theta_deg)))
    ang_mask = np.abs(theta_deg) <= 68.0

    # Single-slit reference run on the snapped mesh (to extract the FIT single-slit envelope)
    single_ref = run_double_slit_simulation(
        slit_width=slit_width,
        slit_sep=3.0 * slit_width,
        wavelength=wavelength,
        screen_thickness=screen_thickness,
        polarization="Hz",
        geometry_mode="single_right",
        cpw=cpw,
        x_half_span=x_span,
        y_upstream=1.2 * wavelength,
        y_downstream=4.1 * wavelength,
        detector_distance=det_dist,
        theta_deg=theta_deg,
        n_cycles=11.5 if quick else 13.5,
    )
    single_env_fit = single_ref["u2d"] / float(single_ref["u2d"][idx0])
    single_env_theory = single_slit_envelope(theta_rad, slit_width, wavelength, obliquity="rs2")

    # Empty-domain normalization run on the d/a = 3 mesh
    empty_ref = run_double_slit_simulation(
        slit_width=slit_width,
        slit_sep=3.0 * slit_width,
        wavelength=wavelength,
        screen_thickness=screen_thickness,
        polarization="Hz",
        geometry_mode="empty",
        cpw=cpw,
        x_half_span=x_span,
        y_upstream=1.2 * wavelength,
        y_downstream=4.1 * wavelength,
        detector_distance=det_dist,
        theta_deg=theta_deg,
        n_cycles=11.5 if quick else 13.5,
    )
    e_inc_det = float(
        np.mean(
            np.abs(
                empty_ref["det_profile"]["field"][
                    np.abs(empty_ref["det_profile"]["x"]) <= 2.5 * wavelength
                ]
            )
        )
    )

    theta_missing_deg = float(np.rad2deg(np.arcsin(wavelength / slit_width)))
    idx_missing = int(np.argmin(np.abs(theta_deg - theta_missing_deg)))

    patterns_fit = []
    patterns_theory = []
    l2_errors = []
    missing_order_intensity = []
    neighbor_order_intensity = []

    map_x = None
    map_y = None
    map_intensity = None
    det_x_over_lam = None
    det_fit_d3 = None
    det_rs1_d3 = None

    for p_int in p_ratios:
        d = float(p_int) * slit_width
        record_map = p_int == 3
        sim = run_double_slit_simulation(
            slit_width=slit_width,
            slit_sep=d,
            wavelength=wavelength,
            screen_thickness=screen_thickness,
            polarization="Hz",
            geometry_mode="double",
            cpw=cpw,
            x_half_span=x_span,
            y_upstream=1.2 * wavelength,
            y_downstream=4.1 * wavelength,
            detector_distance=det_dist,
            record_2d_dft=record_map,
            theta_deg=theta_deg,
            n_cycles=11.5 if quick else 13.5,
        )
        pat_fit = sim["u2d"] / float(sim["u2d"][idx0])
        pat_th = double_slit_farfield_pattern(
            theta_rad, slit_width, d, wavelength, obliquity="rs2"
        )
        err = float(
            np.linalg.norm(pat_fit[ang_mask] - pat_th[ang_mask])
            / np.linalg.norm(pat_th[ang_mask])
        )

        # Intensity at the missing order m = p vs the preceding bright order m = p - 1
        theta_prev_deg = float(np.rad2deg(np.arcsin((p_int - 1) * wavelength / d)))
        idx_prev = int(np.argmin(np.abs(theta_deg - theta_prev_deg)))
        missing_order_intensity.append(float(pat_fit[idx_missing]))
        neighbor_order_intensity.append(float(pat_fit[idx_prev]))

        patterns_fit.append(pat_fit)
        patterns_theory.append(pat_th)
        l2_errors.append(err)

        if record_map:
            map_x = sim["map_2d"]["x"] / wavelength
            map_y = sim["map_2d"]["y"] / wavelength
            map_intensity = (np.abs(sim["map_2d"]["field"]) / e_inc_det) ** 2
            det_x = sim["det_profile"]["x"]
            det_x_over_lam = det_x / wavelength
            det_fit_d3 = (np.abs(sim["det_profile"]["field"]) / e_inc_det) ** 2
            det_rs1_d3 = double_slit_rayleigh_sommerfeld_line(
                det_x, slit_width, d, wavelength, det_dist, kind="II"
            )

    return {
        "slit_width_over_lam": np.array(slit_width / wavelength),
        "theta_missing_deg": np.array(theta_missing_deg),
        "p_ratios": np.asarray(p_ratios, dtype=int),
        "theta_deg": theta_deg,
        "single_env_fit": single_env_fit,
        "single_env_theory": single_env_theory,
        "patterns_fit": np.asarray(patterns_fit),
        "patterns_theory": np.asarray(patterns_theory),
        "l2_errors": np.asarray(l2_errors),
        "missing_order_intensity": np.asarray(missing_order_intensity),
        "neighbor_order_intensity": np.asarray(neighbor_order_intensity),
        "map_x_over_lam": map_x,
        "map_y_over_lam": map_y,
        "map_intensity": map_intensity,
        "det_x_over_lam": det_x_over_lam,
        "det_fit_d3": det_fit_d3,
        "det_rs1_d3": det_rs1_d3,
    }


def run_fringe_spacing_task(quick: bool = False) -> dict:
    """Measure angular fringe spacing Delta(sin theta) = lambda / d across d / lambda."""
    wavelength = DEFAULT_WAVELENGTH
    slit_width = 0.8 * wavelength
    screen_thickness = 0.14 * wavelength
    cpw = 20 if quick else 22
    d_over_lam_list = [1.8, 2.5, 3.2, 4.0]
    det_dist = 3.5 * wavelength

    theta_deg = np.linspace(-78.0, 78.0, 1561)
    theta_rad = np.deg2rad(theta_deg)
    sin_theta = np.sin(theta_rad)
    idx0 = int(np.argmin(np.abs(theta_deg)))

    dsin_fit = []
    dsin_theory = []
    first_peak_deg_fit = []
    first_peak_deg_theory = []
    patterns_fit = []
    patterns_theory = []

    for d_ratio in d_over_lam_list:
        d = float(d_ratio * wavelength)
        sim = run_double_slit_simulation(
            slit_width=slit_width,
            slit_sep=d,
            wavelength=wavelength,
            screen_thickness=screen_thickness,
            polarization="Ez",
            geometry_mode="double",
            cpw=cpw,
            x_half_span=5.5 * wavelength,
            y_upstream=1.2 * wavelength,
            y_downstream=4.0 * wavelength,
            detector_distance=det_dist,
            theta_deg=theta_deg,
            n_cycles=11.5 if quick else 13.5,
        )
        pat_fit = sim["u2d"] / float(sim["u2d"][idx0])
        pat_th = double_slit_farfield_pattern(
            theta_rad, slit_width, d, wavelength, obliquity="rs1"
        )

        # Divide out the smooth single-slit RS-I envelope inside the main lobe to locate
        # the pure m = +1 interference peak at sin(theta_1) = lambda / d
        env_safe = np.maximum(
            single_slit_envelope(theta_rad, slit_width, wavelength, obliquity="rs1"),
            0.05,
        )
        interf_recovered = pat_fit / env_safe
        target_sin = 1.0 / d_ratio
        win = (sin_theta >= 0.65 * target_sin) & (sin_theta <= 1.35 * target_sin)
        sin_win = sin_theta[win]
        theta_win = theta_deg[win]
        imax = int(np.argmax(interf_recovered[win]))

        dsin_meas = float(sin_win[imax])
        dsin_th = fringe_spacing_sin_theta(d, wavelength)
        dsin_fit.append(dsin_meas)
        dsin_theory.append(dsin_th)
        first_peak_deg_fit.append(float(theta_win[imax]))
        first_peak_deg_theory.append(float(np.rad2deg(np.arcsin(dsin_th))))
        patterns_fit.append(pat_fit)
        patterns_theory.append(pat_th)

    return {
        "d_over_lam": np.asarray(d_over_lam_list, dtype=float),
        "slit_width_over_lam": np.array(slit_width / wavelength),
        "theta_deg": theta_deg,
        "sin_theta": sin_theta,
        "dsin_fit": np.asarray(dsin_fit),
        "dsin_theory": np.asarray(dsin_theory),
        "first_peak_deg_fit": np.asarray(first_peak_deg_fit),
        "first_peak_deg_theory": np.asarray(first_peak_deg_theory),
        "patterns_fit": np.asarray(patterns_fit),
        "patterns_theory": np.asarray(patterns_theory),
    }


def run_coupling_task(quick: bool = False) -> dict:
    """Measure inter-slit electromagnetic coupling P'_double(d) / (2 * P'_single) vs d/lambda."""
    wavelength = DEFAULT_WAVELENGTH
    slit_width = 0.55 * wavelength
    screen_thickness = 0.18 * wavelength
    cpw = 20 if quick else 22
    if quick:
        d_over_lam = np.array([0.75, 0.92, 1.12, 1.35, 1.60, 1.90, 2.25], dtype=float)
    else:
        d_over_lam = np.array(
            [0.72, 0.85, 1.00, 1.15, 1.32, 1.50, 1.70, 1.95, 2.25], dtype=float
        )

    x_span = 4.6 * wavelength

    # Single-slit reference transmitted power for E_z and H_z on the snapped slit mesh
    single_ez = run_double_slit_simulation(
        slit_width=slit_width,
        slit_sep=1.50 * wavelength,
        wavelength=wavelength,
        screen_thickness=screen_thickness,
        polarization="Ez",
        geometry_mode="single_right",
        cpw=cpw,
        min_slit_cells=6,
        min_screen_cells=4,
        x_half_span=x_span,
        y_upstream=1.2 * wavelength,
        y_downstream=2.0 * wavelength,
        detector_distance=None,
        n_cycles=11.5 if quick else 13.0,
    )
    single_hz = run_double_slit_simulation(
        slit_width=slit_width,
        slit_sep=1.50 * wavelength,
        wavelength=wavelength,
        screen_thickness=screen_thickness,
        polarization="Hz",
        geometry_mode="single_right",
        cpw=cpw,
        min_slit_cells=6,
        min_screen_cells=4,
        x_half_span=x_span,
        y_upstream=1.2 * wavelength,
        y_downstream=2.0 * wavelength,
        detector_distance=None,
        n_cycles=11.5 if quick else 13.0,
    )
    p1_ez = single_ez["p_n2f_per_z"]
    p1_hz = single_hz["p_n2f_per_z"]

    eta_ez = []
    eta_hz = []
    bridge_over_lam = d_over_lam - (slit_width / wavelength)

    for d_ratio in d_over_lam:
        d = float(d_ratio * wavelength)
        sim_ez = run_double_slit_simulation(
            slit_width=slit_width,
            slit_sep=d,
            wavelength=wavelength,
            screen_thickness=screen_thickness,
            polarization="Ez",
            geometry_mode="double",
            cpw=cpw,
            min_slit_cells=6,
            min_barrier_cells=4,
            min_screen_cells=4,
            x_half_span=x_span,
            y_upstream=1.2 * wavelength,
            y_downstream=2.0 * wavelength,
            detector_distance=None,
            n_cycles=11.5 if quick else 13.0,
        )
        sim_hz = run_double_slit_simulation(
            slit_width=slit_width,
            slit_sep=d,
            wavelength=wavelength,
            screen_thickness=screen_thickness,
            polarization="Hz",
            geometry_mode="double",
            cpw=cpw,
            min_slit_cells=6,
            min_barrier_cells=4,
            min_screen_cells=4,
            x_half_span=x_span,
            y_upstream=1.2 * wavelength,
            y_downstream=2.0 * wavelength,
            detector_distance=None,
            n_cycles=11.5 if quick else 13.0,
        )
        eta_ez.append(sim_ez["p_n2f_per_z"] / (2.0 * p1_ez))
        eta_hz.append(sim_hz["p_n2f_per_z"] / (2.0 * p1_hz))

    fine_d = np.linspace(0.68, 2.30, 168)
    models = surface_wave_coupling_model(fine_d)

    return {
        "slit_width_over_lam": np.array(slit_width / wavelength),
        "screen_thickness_over_lam": np.array(screen_thickness / wavelength),
        "d_over_lam": d_over_lam,
        "bridge_over_lam": bridge_over_lam,
        "eta_ez": np.asarray(eta_ez),
        "eta_hz": np.asarray(eta_hz),
        "model_d_over_lam": fine_d,
        "model_eta_ez": models["eta_ez_model"],
        "model_eta_hz": models["eta_hz_model"],
    }


def run_resolution_task(quick: bool = False) -> dict:
    """Grid convergence study over cpw for d/a = 3 (a = 1.0*lambda, d = 3.0*lambda)."""
    wavelength = DEFAULT_WAVELENGTH
    slit_width = 1.0 * wavelength
    slit_sep = 3.0 * wavelength
    screen_thickness = 0.15 * wavelength
    cpw_list = [14, 20, 28] if quick else [12, 16, 20, 28, 36]

    theta_deg = np.linspace(-80.0, 80.0, 641)
    theta_rad = np.deg2rad(theta_deg)
    idx0 = int(np.argmin(np.abs(theta_deg)))
    ang_mask = np.abs(theta_deg) <= 65.0
    theory_ref = double_slit_farfield_pattern(
        theta_rad, slit_width, slit_sep, wavelength, obliquity="rs2"
    )

    l2_vs_theory = []
    patterns = []
    cells_slit_list = []
    cells_bridge_list = []
    cells_screen_list = []
    elapsed_list = []

    for cpw in cpw_list:
        sc = max(4, int(round(1.0 * cpw)))
        bc = max(4, int(round(2.0 * cpw)))
        tc = max(3, int(round(0.15 * cpw)))
        sim = run_double_slit_simulation(
            slit_width=slit_width,
            slit_sep=slit_sep,
            wavelength=wavelength,
            screen_thickness=screen_thickness,
            polarization="Hz",
            geometry_mode="double",
            cpw=cpw,
            min_slit_cells=sc,
            min_barrier_cells=bc,
            min_screen_cells=tc,
            x_half_span=5.2 * wavelength,
            y_upstream=1.2 * wavelength,
            y_downstream=2.0 * wavelength,
            detector_distance=None,
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


def _build_schematic_solver(
    slit_width: float = 1.0 * DEFAULT_WAVELENGTH,
    slit_sep: float = 3.0 * DEFAULT_WAVELENGTH,
    wavelength: float = DEFAULT_WAVELENGTH,
    screen_thickness: float = 0.15 * DEFAULT_WAVELENGTH,
    cpw: int = 22,
    x_half_span: float | None = None,
    y_upstream: float = 1.2 * DEFAULT_WAVELENGTH,
    y_downstream: float = 4.0 * DEFAULT_WAVELENGTH,
    detector_distance: float = 3.5 * DEFAULT_WAVELENGTH,
) -> tuple[SolverFIT3D, dict, list[Box], CurrentSource, list]:
    """Build a lightweight 2D double-slit solver, PEC boxes, source, and monitors for schematics/overlays."""
    f0 = c0 / wavelength
    grid_info = build_double_slit_grid(
        slit_width=slit_width,
        slit_sep=slit_sep,
        wavelength=wavelength,
        screen_thickness=screen_thickness,
        x_half_span=x_half_span,
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
    x_outer = 0.5 * slit_sep + 0.5 * slit_width
    x_inner = 0.5 * slit_sep - 0.5 * slit_width
    w_left = (-x_outer) - left_end
    xc_left = 0.5 * (left_end - x_outer)
    w_right = right_end - x_outer
    xc_right = 0.5 * (x_outer + right_end)
    w_bridge = 2.0 * x_inner
    boxes = [
        Box((xc_left, 0.0, 0.0), (w_left, screen_thickness, 2.0 * lz), "pec"),
        Box((0.0, 0.0, 0.0), (w_bridge, screen_thickness, 2.0 * lz), "pec"),
        Box((xc_right, 0.0, 0.0), (w_right, screen_thickness, 2.0 * lz), "pec"),
    ]
    for b in boxes:
        b.material = "pec"
    assign_materials(solver, boxes, f0)

    y_src = float(y_core[0] + 0.35 * wavelength)
    src_span_x = float(x[-1] - x[0])
    pulse = GaussianSource(frequency=f0, fwidth=0.5 * f0, cutoff=3.5)
    source = CurrentSource(
        src_time=pulse,
        component="x",
        center=(0.0, y_src, 0.0),
        size=(src_span_x, 0.0, 2.0 * lz),
        amplitude=1.0,
    )

    freqs = np.array([f0])
    w_ref = float((x_core[-1] - x_core[0]) - 0.5 * wavelength)
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
    mon_det = DFTMonitor(
        frequencies=freqs,
        components=("Hz",),
        center=(0.0, float(half_t + detector_distance), 0.0),
        size=(w_ref, 0.0, 0.0),
    )
    return solver, grid_info, boxes, source, [flux_wide, n2f, mon_det]


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


def _reposition_xy_schematic_labels(
    ax: plt.Axes,
    x_left_jaw: float,
    x_right_jaw: float,
    y_src: float,
    y_n2f: float,
    y_dft: float,
) -> None:
    """Reposition horizontal structure/source/monitor labels placed at vertical-slab defaults by schematic."""
    leg = ax.get_legend()
    if leg is not None:
        leg.remove()
    pec_count = 0
    pec_positions = (x_left_jaw, 0.0, x_right_jaw)
    pec_labels = (r"PEC ($\sigma \to \infty$)", "PEC bridge $b$", r"PEC ($\sigma \to \infty$)")
    for txt in ax.texts:
        s = txt.get_text()
        if "$-y$ boundary:" in s or "-y boundary:" in s:
            txt.set_text(
                r"$\pm y$ boundaries: CPML (10 cells, $0.27\;\mu\mathrm{m}$) $\mid$ Out-of-plane ($\pm z$: Periodic)"
            )
        elif "PEC" in s:
            idx = min(pec_count, 2)
            txt.set_text(pec_labels[idx])
            txt.set_position((pec_positions[idx], -0.15))
            txt.set_ha("center")
            txt.set_va("top")
            pec_count += 1
        elif "CurrentSource" in s:
            txt.set_position((0.0, y_src - 0.12))
            txt.set_ha("center")
            txt.set_va("top")
        elif "Near2Far" in s:
            txt.set_text("Near2FarMonitor + FluxMonitor")
            txt.set_position((0.0, y_n2f + 0.12))
            txt.set_ha("center")
            txt.set_va("bottom")
        elif "FluxMonitor" in s:
            txt.set_text("")
        elif "DFTMonitor" in s:
            txt.set_text(r"DFTMonitor ($L = 3.5\lambda$)")
            txt.set_position((0.0, y_dft + 0.10))
            txt.set_ha("center")
            txt.set_va("bottom")
    _fix_unit_spaces_in_texts(ax)


def plot_setup_schematic(
    output_path: str | Path,
    wavelength: float = DEFAULT_WAVELENGTH,
) -> None:
    """Generate the 2-panel labelled setup schematic and double-slit mesh zoom (Figure 0)."""
    slit_width = 1.0 * wavelength
    slit_sep = 3.0 * wavelength
    screen_thickness = 0.15 * wavelength
    cpw = 22
    solver, grid_info, boxes, source, monitors = _build_schematic_solver(
        slit_width=slit_width,
        slit_sep=slit_sep,
        wavelength=wavelength,
        screen_thickness=screen_thickness,
        cpw=cpw,
    )
    x_um = grid_info["x"] * 1e6
    y_core_um = grid_info["y_core"] * 1e6
    a_um = slit_width * 1e6
    d_um = slit_sep * 1e6
    b_um = (slit_sep - slit_width) * 1e6
    t_um = screen_thickness * 1e6
    x_outer_um = 0.5 * (d_um + a_um)
    y_src_um = float(y_core_um[0] + 0.35 * wavelength * 1e6)
    y_n2f_um = float(0.5 * t_um + 2.5 * grid_info["dy_screen"] * 1e6)
    y_det_um = float(0.5 * t_um + 3.5 * wavelength * 1e6)

    with figstyle.apply_style():
        fig, axes = figstyle.new_figure(nrows=1, ncols=2, width=13.8, height=5.0)

        # Panel (a): Full 2D Young's double-slit setup schematic
        ax = axes[0]
        schematic.plot_schematic(
            solver,
            ax=ax,
            plane="xy",
            unit="µm",
            structures=boxes,
            sources=[source],
            monitors=monitors,
            wavelength=wavelength,
            show_mesh=False,
            mesh_zoom=False,
            show_wave_packet=False,
            title=r"(a) Young's Double-Slit Setup ($\lambda = 0.60\;\mu\mathrm{m}$, $a = \lambda$, $d = 3\lambda$)",
        )
        _reposition_xy_schematic_labels(
            ax,
            x_left_jaw=0.5 * (float(x_um[0]) - x_outer_um),
            x_right_jaw=0.5 * (x_outer_um + float(x_um[-1])),
            y_src=y_src_um,
            y_n2f=y_n2f_um,
            y_dft=y_det_um,
        )
        # Dimension arrow for center-to-center separation d
        ax.annotate(
            "",
            xy=(0.5 * d_um, 0.90),
            xytext=(-0.5 * d_um, 0.90),
            arrowprops={"arrowstyle": "<->", "color": figstyle.INK, "lw": 1.2},
            zorder=9,
        )
        ax.text(
            0.0,
            0.96,
            rf"Slit pitch $d = {d_um:.2f}\;\mu\mathrm{{m}}$ ($3\lambda$)",
            ha="center",
            va="bottom",
            fontsize=8.4,
            fontweight="bold",
            color=figstyle.INK,
            zorder=9,
        )

        # Panel (b): Zoomed graded FIT mesh around both slits and the central PEC bridge
        ax_m = axes[1]
        ax_m.set_facecolor("#F8FAFC")
        schematic.draw_mesh(
            ax_m,
            solver,
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
            solver,
            plane="xy",
            unit="µm",
            structures=boxes,
            show_boundaries=False,
            show_materials=True,
            fill_materials=True,
            material_alpha=0.55,
            outline_color=figstyle.INK,
            label_materials=False,
        )
        ax_m.set_xlim(-1.60, 1.60)
        ax_m.set_ylim(-0.34, 0.34)
        ax_m.set_aspect("auto")
        ax_m.set_xlabel("x [µm]")
        ax_m.set_ylabel("y [µm]")
        ax_m.set_title(r"(b) Snapped Mesh Zoom at Slits ($\pm d/2$) and PEC Bridge ($b = d - a$)")

        # Annotate central PEC bridge and slits
        ax_m.text(
            0.0,
            0.0,
            f"PEC bridge\n$b = {b_um:.2f}$ µm ({grid_info['cells_across_bridge']} cells $\\geq 4$)",
            ha="center",
            va="center",
            fontsize=8.0,
            fontweight="bold",
            color=figstyle.INK,
            bbox={
                "boxstyle": "round,pad=0.16",
                "facecolor": figstyle.PAPER,
                "edgecolor": "#94A3B8",
                "alpha": 0.92,
            },
            zorder=9,
        )
        for sign, x_c in ((-1, -0.5 * d_um), (1, 0.5 * d_um)):
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
                f"$a = {a_um:.2f}$ µm\n({grid_info['cells_across_slit']} cells $\\geq 4$)",
                ha="center",
                va="bottom",
                fontsize=8.0,
                fontweight="bold",
                color="#0072B2",
                zorder=9,
            )
        ax_m.annotate(
            "",
            xy=(-1.32, 0.5 * t_um),
            xytext=(-1.32, -0.5 * t_um),
            arrowprops={"arrowstyle": "<->", "color": "#D55E00", "lw": 1.3},
            zorder=9,
        )
        ax_m.text(
            -1.32,
            -0.14,
            f"Screen $t = {t_um*1e3:.0f}$ nm ({grid_info['cells_across_screen']} cells $\\geq 4$)",
            ha="left",
            va="top",
            fontsize=8.3,
            fontweight="bold",
            color="#D55E00",
            zorder=9,
        )
        _fix_unit_spaces_in_texts(ax_m)

        figstyle.save(fig, output_path, close=True)


def plot_missing_orders_and_nearfield(
    mo_data: dict,
    output_path: str | Path,
) -> None:
    """Generate the 3-panel figure for 2D interference map, screen fringes, and missing orders."""
    wavelength = DEFAULT_WAVELENGTH
    lam_um = wavelength * 1e6
    a_mo = float(mo_data.get("slit_width_over_lam", 1.5)) * wavelength
    d_mo = 3.0 * a_mo
    t_mo = 0.14 * wavelength
    solver_mo, _, boxes_mo, _, _ = _build_schematic_solver(
        slit_width=a_mo,
        slit_sep=d_mo,
        wavelength=wavelength,
        screen_thickness=t_mo,
        cpw=22,
        x_half_span=6.2 * wavelength,
        y_upstream=1.2 * wavelength,
        y_downstream=4.1 * wavelength,
    )

    with figstyle.apply_style():
        fig, axes = figstyle.new_figure(nrows=1, ncols=3, width=15.0, height=4.8)

        # Panel (a): 2D intensity map |H_z(x, y)|^2 / |H_inc|^2 for d/a = 3 (a = 1.5*lambda, d = 4.5*lambda)
        ax = axes[0]
        x_um = mo_data["map_x_over_lam"] * lam_um
        y_um = mo_data["map_y_over_lam"] * lam_um
        inten = mo_data["map_intensity"]
        im = ax.pcolormesh(
            x_um,
            y_um,
            inten.T,
            shading="auto",
            cmap=figstyle.CMAP_INTENSITY,
            vmin=0.0,
            vmax=min(float(np.percentile(inten, 99.5)), 2.8),
        )
        fig.colorbar(im, ax=ax, label=r"$|H_z(x, y)|^2 / |H_{\mathrm{inc}}|^2$ [-]")
        schematic.overlay_geometry(
            ax,
            solver_mo,
            plane="xy",
            unit="µm",
            orientation="2d",
            structures=boxes_mo,
            show_boundaries=True,
            show_materials=True,
            fill_materials=True,
            material_alpha=0.38,
            outline_color="#FFFFFF",
            label_materials=False,
        )
        ax.text(
            0.0,
            0.14,
            "PEC bridge",
            color="white",
            fontsize=7.6,
            fontweight="bold",
            ha="center",
            va="bottom",
            zorder=8,
        )
        y_det_um = 3.5 * lam_um
        ax.axhline(y_det_um, color="#009E73", ls="--", lw=1.2, zorder=7)
        ax.text(
            float(x_um[-1]) - 0.15,
            y_det_um - 0.22,
            rf"Detector $L = 3.5\lambda$ ({y_det_um:.1f} µm)",
            color="white",
            fontsize=7.6,
            fontweight="bold",
            ha="right",
            zorder=8,
        )
        ax.set_xlim(float(x_um[0]), float(x_um[-1]))
        ax.set_ylim(float(y_um[0]), float(y_um[-1]))
        ax.set_aspect("auto")
        ax.set_xlabel(r"Transverse coordinate $x$ [µm]")
        ax.set_ylabel(r"Propagation coordinate $y$ [µm]")
        ax.set_title(r"(a) Young's double-slit near field ($a = 1.5\lambda$, $d = 4.5\lambda$)")

        # Panel (b): Transverse detector line profile at L = 3.5 * lambda vs 2D Rayleigh-Sommerfeld II
        ax = axes[1]
        ax.plot(
            mo_data["det_x_over_lam"],
            mo_data["det_fit_d3"],
            **figstyle.series_style(0, with_marker=False, lw=2.1),
            label=r"2D FIT detector line ($L = 3.5\lambda$)",
        )
        ax.plot(
            mo_data["det_x_over_lam"],
            mo_data["det_rs1_d3"],
            **figstyle.series_style(1, with_marker=False, lw=1.6),
            label=r"2D Rayleigh–Sommerfeld II integral",
        )
        ax.set_xlim(-5.5, 5.5)
        ax.set_ylim(-0.03, 1.05)
        ax.set_xlabel(r"Transverse detector coordinate $x / \lambda$ [-]")
        ax.set_ylabel(r"Normalized intensity $|H_z(x, L)|^2 / |H_{\mathrm{inc}}|^2$ [-]")
        ax.set_title(r"(b) Finite-distance screen profile at $L = 3.5\lambda$ ($d/a = 3$)")
        figstyle.spectrum_grid(ax)
        ax.legend(loc="upper right", fontsize=7.8)

        # Panel (c): Far-field cos^2 * sinc^2 envelope and missing orders for d/a in {2, 3, 4}
        ax = axes[2]
        theta_deg = mo_data["theta_deg"]
        theta_miss = float(mo_data["theta_missing_deg"])
        env_fit = mo_data["single_env_fit"]
        p_ratios = mo_data["p_ratios"]

        offsets = [0.0, 1.15, 2.30]
        for k in range(len(p_ratios)):
            p_int = int(p_ratios[k])
            off = offsets[k]
            err = float(mo_data["l2_errors"][k]) * 100.0
            ax.plot(
                theta_deg,
                mo_data["patterns_fit"][k] + off,
                color=figstyle.PALETTE[k % len(figstyle.PALETTE)],
                lw=2.0,
                ls="-",
                label=rf"FIT $d/a={p_int}$ ($m=\pm{p_int}$, $L^2={err:.1f}\%$)",
            )
            ax.plot(
                theta_deg,
                mo_data["patterns_theory"][k] + off,
                color=figstyle.INK,
                lw=1.2,
                ls="--",
                label=r"Theory $\cos^2\alpha\,\mathrm{sinc}^2\beta$" if k == 0 else "_nolegend_",
            )
            ax.plot(
                theta_deg,
                env_fit + off,
                color="#D55E00",
                lw=1.3,
                ls=":",
                label="FIT single-slit envelope" if k == 0 else "_nolegend_",
            )
        ax.axvline(theta_miss, color="#CC79A7", ls="-.", lw=1.2)
        ax.axvline(
            -theta_miss,
            color="#CC79A7",
            ls="-.",
            lw=1.2,
            label=rf"Missing order $\pm {theta_miss:.1f}^\circ$",
        )
        ax.set_xlim(-75.0, 75.0)
        ax.set_ylim(-0.05, 4.10)
        ax.set_xlabel(r"Diffraction angle $\theta$ [deg]")
        ax.set_ylabel(
            r"Normalized far-field intensity $U_{\mathrm{2D}}(\theta)/U_{\mathrm{2D}}(0)$ (offset) [-]"
        )
        ax.set_title(r"(c) Far-field missing orders ($d/a = 2, 3, 4$)")
        figstyle.spectrum_grid(ax)
        ax.legend(loc="upper center", ncol=2, fontsize=6.6)

        figstyle.save(fig, output_path, close=True)


def plot_fringe_spacing_scaling(
    fringe_data: dict,
    output_path: str | Path,
) -> None:
    """Generate the 2-panel figure for fringe spacing Delta(sin theta) = lambda / d."""
    with figstyle.apply_style():
        fig, axes = figstyle.new_figure(nrows=1, ncols=2, width=13.8, height=4.8)

        # Panel (a): Normalized patterns vs sin(theta) showing uniform period lambda / d
        ax = axes[0]
        sin_th = fringe_data["sin_theta"]
        d_ratios = fringe_data["d_over_lam"]
        for k, d_ratio in enumerate(d_ratios):
            off = k * 1.10
            ax.plot(
                sin_th,
                fringe_data["patterns_fit"][k] + off,
                color=figstyle.PALETTE[k % len(figstyle.PALETTE)],
                lw=2.0,
                ls="-",
                label=rf"FIT $d/\lambda = {d_ratio:g}$ ($\Delta\sin\theta = {fringe_data['dsin_fit'][k]:.3f}$)",
            )
            ax.plot(
                sin_th,
                fringe_data["patterns_theory"][k] + off,
                color=figstyle.INK,
                lw=1.2,
                ls="--",
                label="Theory" if k == 0 else "_nolegend_",
            )
        ax.set_xlim(-0.92, 0.92)
        ax.set_ylim(-0.05, 5.15)
        ax.set_xlabel(r"Direction cosine $\sin\theta$ [-]")
        ax.set_ylabel(r"Normalized far-field intensity (offset) [-]")
        ax.set_title(r"(a) Interference fringes in $\sin\theta$ space ($a = 0.8\lambda$)")
        figstyle.spectrum_grid(ax)
        ax.legend(loc="upper center", ncol=2, fontsize=7.0)

        # Panel (b): Extracted fringe spacing Delta(sin theta) and first-order peak angle vs d / lambda
        ax = axes[1]
        d_fine = np.linspace(1.5, 4.3, 141)
        ax.plot(
            d_fine,
            1.0 / d_fine,
            color=figstyle.INK,
            lw=1.8,
            ls="--",
            label=r"Exact grating law $\Delta(\sin\theta) = \lambda / d$",
        )
        ax.plot(
            d_ratios,
            fringe_data["dsin_fit"],
            "o",
            color=figstyle.PALETTE[0],
            ms=6.5,
            label=r"2D FIT Near2FarMonitor measured $\Delta(\sin\theta)$",
        )
        ax2 = ax.twinx()
        ax2.plot(
            d_fine,
            np.rad2deg(np.arcsin(1.0 / d_fine)),
            color=figstyle.PALETTE[1],
            lw=1.6,
            ls=":",
            label=r"Theory $\theta_1 = \arcsin(\lambda / d)$ [deg]",
        )
        ax2.plot(
            d_ratios,
            fringe_data["first_peak_deg_fit"],
            "s",
            color=figstyle.PALETTE[1],
            ms=6.0,
            label=r"FIT measured $\theta_1$ [deg]",
        )
        ax.set_xlabel(r"Normalized slit separation $d / \lambda$ [-]")
        ax.set_ylabel(r"Fringe spacing $\Delta(\sin\theta)$ [-]", color=figstyle.PALETTE[0])
        ax2.set_ylabel(r"First-order maximum angle $\theta_1$ [deg]", color=figstyle.PALETTE[1])
        ax.set_title(r"(b) Fringe spacing scaling $\Delta(\sin\theta) = \lambda / d$")
        figstyle.spectrum_grid(ax)
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=7.8)

        figstyle.save(fig, output_path, close=True)


def plot_coupling_and_resolution(
    coup_data: dict,
    res_data: dict,
    output_path: str | Path,
) -> None:
    """Generate the 2-panel figure for inter-slit coupling and mesh convergence."""
    with figstyle.apply_style():
        fig, axes = figstyle.new_figure(nrows=1, ncols=2, width=13.8, height=4.8)

        # Panel (a): Normalized mutual transmission coupling eta = P'_double(d) / (2 * P'_single)
        ax = axes[0]
        d_lam = coup_data["d_over_lam"]
        ax.plot(
            d_lam,
            coup_data["eta_hz"],
            "s-",
            color=figstyle.PALETTE[1],
            lw=2.0,
            ms=5.5,
            label=r"FIT $\mathbf{E} \perp$ slit ($H_z$, TEM + surface wave)",
        )
        ax.plot(
            d_lam,
            coup_data["eta_ez"],
            "o-",
            color=figstyle.PALETTE[0],
            lw=2.0,
            ms=5.5,
            label=r"FIT $\mathbf{E} \parallel$ slit ($E_z$, Dirichlet screen)",
        )
        ax.plot(
            coup_data["model_d_over_lam"],
            coup_data["model_eta_hz"],
            color="#8C2D04",
            ls=":",
            lw=1.5,
            label=r"Cylindrical surface-wave model $\propto \cos(k_0 d + \phi)/\sqrt{d/\lambda}$",
        )
        ax.axhline(
            1.0,
            color="#64748B",
            ls="--",
            lw=1.1,
            label=r"Uncoupled slits $P'_{\mathrm{double}} = 2 P'_{\mathrm{single}}$",
        )
        ax.set_xlabel(r"Center-to-center slit separation $d / \lambda$ [-]")
        ax.set_ylabel(
            r"Normalized transmission ratio $P'_{\mathrm{double}}(d) / (2 P'_{\mathrm{single}})$ [-]"
        )
        ax.set_title(
            r"(a) Inter-slit electromagnetic coupling ($a = 0.55\lambda$, $t = 0.18\lambda$)"
        )
        figstyle.spectrum_grid(ax)
        ax.legend(loc="upper right", fontsize=7.6)

        # Panel (b): Mesh resolution convergence study
        ax = axes[1]
        cpw = res_data["cpw_values"]
        ax.plot(
            cpw,
            res_data["l2_vs_theory"] * 100.0,
            "o-",
            color=figstyle.PALETTE[0],
            lw=2.0,
            ms=5.5,
            label=r"Far-field $L^2$ error vs $\cos^2\alpha\,\mathrm{sinc}^2\beta$ [%]",
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
        ax.set_title(r"(b) Mesh convergence ($a = \lambda$, $d = 3\lambda$, $d/a = 3$)")
        figstyle.spectrum_grid(ax)
        ax.legend(loc="upper right", fontsize=8.0)

        figstyle.save(fig, output_path, close=True)
