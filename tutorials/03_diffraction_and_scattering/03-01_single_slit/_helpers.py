"""Helper functions for Tutorial 3.1: Diffraction by a Single Slit."""

from __future__ import annotations

from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c as c0
from scipy.special import fresnel, hankel1

from wakis import DFTMonitor, FieldMonitor, FluxMonitor, GridFIT3D, SolverFIT3D, figstyle, schematic
from wakis.conformal import rectangle_sheet
from wakis.estimate import estimate, preflight
from wakis.near2far import Near2FarMonitor
from wakis.sources import CurrentSource, GaussianSource
from wakis.templates.meshing import graded_axis
from wakis.templates.shapes import Box, assign_materials

DEFAULT_WAVELENGTH = 0.6e-6  # 600 nm (500 THz)


def sinc_phys(u: np.ndarray | float) -> np.ndarray:
    """Return the unnormalized sinc function sin(u) / u.

    Parameters
    ----------
    u : array_like
        Dimensionless argument [rad].

    Returns
    -------
    ndarray
        Values of sin(u) / u (1 at u = 0) [-].
    """
    u_arr = np.asarray(u, dtype=float)
    return np.sinc(u_arr / np.pi)


def fraunhofer_pattern(
    theta: np.ndarray,
    slit_width: float,
    wavelength: float = DEFAULT_WAVELENGTH,
) -> np.ndarray:
    """Normalized Fraunhofer far-field intensity pattern sinc^2(beta) for a single slit.

    Parameters
    ----------
    theta : ndarray
        Diffraction angle measured from the forward (+y) normal [rad].
    slit_width : float
        Slit opening width a [m].
    wavelength : float, optional
        Free-space wavelength lambda [m].

    Returns
    -------
    ndarray
        Normalized angular intensity I(theta) / I(0) [-].
    """
    theta_arr = np.asarray(theta, dtype=float)
    beta = np.pi * (slit_width / wavelength) * np.sin(theta_arr)
    return sinc_phys(beta) ** 2


def kirchhoff_obliquity_pattern(
    theta: np.ndarray,
    slit_width: float,
    wavelength: float = DEFAULT_WAVELENGTH,
) -> np.ndarray:
    """Normalized Kirchhoff far-field intensity pattern with ((1 + cos(theta))/2)^2 obliquity.

    Parameters
    ----------
    theta : ndarray
        Diffraction angle measured from the forward (+y) normal [rad].
    slit_width : float
        Slit opening width a [m].
    wavelength : float, optional
        Free-space wavelength lambda [m].

    Returns
    -------
    ndarray
        Normalized angular intensity I(theta) / I(0) [-].
    """
    theta_arr = np.asarray(theta, dtype=float)
    obliquity = 0.5 * (1.0 + np.cos(theta_arr))
    return (obliquity**2) * fraunhofer_pattern(theta_arr, slit_width, wavelength)


def rayleigh_sommerfeld_farfield(
    theta: np.ndarray,
    slit_width: float,
    wavelength: float = DEFAULT_WAVELENGTH,
    kind: str = "I",
) -> np.ndarray:
    """Normalized 2D Rayleigh-Sommerfeld far-field angular intensity pattern.

    Under the Smythe-Kirchhoff aperture approximation (unperturbed incident field
    in the slit), the 2D Rayleigh-Sommerfeld I integral (Dirichlet screen, E || slit
    edges) carries a cos^2(theta) obliquity factor, while the 2D Rayleigh-Sommerfeld II
    integral (Neumann screen, E perp slit edges / H || slit edges) carries unit
    obliquity factor.

    Parameters
    ----------
    theta : ndarray
        Diffraction angle measured from the forward (+y) normal [rad].
    slit_width : float
        Slit opening width a [m].
    wavelength : float, optional
        Free-space wavelength lambda [m].
    kind : {"I", "II"}, optional
        Rayleigh-Sommerfeld formulation: "I" (Dirichlet / E_z) or "II" (Neumann / H_z).

    Returns
    -------
    ndarray
        Normalized angular intensity I(theta) / I(0) [-].
    """
    theta_arr = np.asarray(theta, dtype=float)
    base = fraunhofer_pattern(theta_arr, slit_width, wavelength)
    if kind.upper() == "I":
        return (np.cos(theta_arr) ** 2) * base
    if kind.upper() == "II":
        return base
    raise ValueError(f"Unknown Rayleigh-Sommerfeld kind {kind!r}; expected 'I' or 'II'.")


def fresnel_number(
    slit_width: float,
    wavelength: float,
    distance: float,
) -> float:
    """Compute the slit Fresnel number N_F = a^2 / (4 * lambda * L).

    Parameters
    ----------
    slit_width : float
        Slit width a [m].
    wavelength : float
        Free-space wavelength lambda [m].
    distance : float
        Observation distance L downstream of the screen [m].

    Returns
    -------
    float
        Dimensionless Fresnel number N_F [-].
    """
    if distance <= 0.0 or wavelength <= 0.0 or slit_width < 0.0:
        raise ValueError("slit_width >= 0, wavelength > 0, and distance > 0 are required.")
    return float((slit_width**2) / (4.0 * wavelength * distance))


def fresnel_cornu_profile(
    x: np.ndarray,
    slit_width: float,
    wavelength: float,
    distance: float,
) -> np.ndarray:
    """Scalar Fresnel near-field intensity I(x, L) / I_inc via Cornu-spiral integrals.

    Parameters
    ----------
    x : ndarray
        Transverse coordinate across the observation plane [m].
    slit_width : float
        Slit opening width a [m].
    wavelength : float
        Free-space wavelength lambda [m].
    distance : float
        Downstream propagation distance L from the aperture plane [m].

    Returns
    -------
    ndarray
        Normalized intensity |E(x, L)|^2 / |E_inc|^2 [-].
    """
    x_arr = np.asarray(x, dtype=float)
    scale = np.sqrt(2.0 / (wavelength * distance))
    u1 = scale * (x_arr - 0.5 * slit_width)
    u2 = scale * (x_arr + 0.5 * slit_width)
    s1, c1 = fresnel(u1)
    s2, c2 = fresnel(u2)
    return 0.5 * ((c2 - c1) ** 2 + (s2 - s1) ** 2)


def rayleigh_sommerfeld_line_profile(
    x: np.ndarray,
    slit_width: float,
    wavelength: float,
    distance: float,
    kind: str = "I",
    n_quad: int = 401,
) -> np.ndarray:
    """Exact 2D Rayleigh-Sommerfeld intensity profile on a line at distance L.

    Evaluates the 2D cylindrical-wave Green's function integral over the slit
    x' in [-a/2, +a/2] assuming an unperturbed unit-amplitude normal-incidence
    plane wave in the aperture.

    Parameters
    ----------
    x : ndarray
        Transverse observation coordinates [m].
    slit_width : float
        Slit opening width a [m].
    wavelength : float
        Free-space wavelength lambda [m].
    distance : float
        Downstream distance L from the slit exit plane [m].
    kind : {"I", "II"}, optional
        "I" for Dirichlet (E_z) or "II" for Neumann (H_z).
    n_quad : int, optional
        Number of Gauss-Legendre quadrature nodes across the slit [-].

    Returns
    -------
    ndarray
        Normalized intensity |psi(x, L)|^2 / |psi_inc|^2 [-].
    """
    x_arr = np.asarray(x, dtype=float)
    k0 = 2.0 * np.pi / wavelength
    xi, wi = np.polynomial.legendre.leggauss(int(n_quad))
    x_prime = 0.5 * slit_width * xi
    weights = 0.5 * slit_width * wi

    dx = x_arr[:, None] - x_prime[None, :]
    r = np.hypot(dx, distance)
    kr = k0 * r

    if kind.upper() == "I":
        kernel = (k0 * distance / (2.0j)) * (hankel1(1, kr) / r)
    elif kind.upper() == "II":
        kernel = 0.5 * k0 * hankel1(0, kr)
    else:
        raise ValueError(f"Unknown Rayleigh-Sommerfeld kind {kind!r}; expected 'I' or 'II'.")

    field = kernel @ weights
    return np.abs(field) ** 2


def fabry_perot_slit_transmission_models(
    a_over_lam: np.ndarray,
    t_over_lam: float,
) -> dict[str, np.ndarray]:
    """Analytical single-mode waveguide approximation for slit transmission tau(a/lambda).

    For E || slit edges (TE_10 slab waveguide mode, E_z), the fundamental mode has a hard
    cutoff at a/lambda = 0.5; below cutoff, propagation constant k_y = i*gamma is purely
    imaginary and transmission decays as exp(-2 * gamma * t).
    For E perp slit edges (TEM parallel-plate mode, H_z), there is no cutoff (k_y = k_0
    for all a > 0) and Fabry-Perot reflections between the entrance and exit apertures
    yield resonant transmission peaks (Takakura, Phys. Rev. Lett. 86, 5601, 2001).

    Parameters
    ----------
    a_over_lam : ndarray
        Normalized slit widths a / lambda [-].
    t_over_lam : float
        Normalized screen thickness t / lambda [-].

    Returns
    -------
    dict of str to ndarray
        Dictionary with keys ``"tau_Ez_model"`` and ``"tau_Hz_model"`` [-].
    """
    ratio = np.asarray(a_over_lam, dtype=float)
    k0_a = 2.0 * np.pi * ratio
    k0_t = 2.0 * np.pi * float(t_over_lam)

    # TEM mode (H_z, E perp slit edges): Takakura single-mode admittance model
    euler_gamma = 0.5772156649015329
    g_rad = 0.5 * k0_a
    b_rad = (k0_a / np.pi) * np.log(np.pi * np.e / np.maximum(euler_gamma * k0_a, 1e-6))
    y_norm = g_rad + 1j * b_rad
    r_tem = (1.0 - y_norm) / (1.0 + y_norm)
    phase_tem = np.exp(1j * k0_t)
    tau_hz = (
        np.abs((1.0 + r_tem) * (1.0 - r_tem) * phase_tem / (1.0 - (r_tem**2) * (phase_tem**2)))
        ** 2
        / np.maximum(g_rad, 1e-6)
    )
    # Blend smoothly toward geometrical optics (tau -> 1) above a/lambda ~ 0.6
    blend = np.clip((ratio - 0.35) / 0.45, 0.0, 1.0)
    tau_hz = (1.0 - blend) * tau_hz + blend * 1.0

    # TE_1 mode (E_z, E || slit edges): cutoff at a/lambda = 0.5
    safe_ratio = np.where(np.abs(ratio - 0.5) < 1e-6, ratio + 1e-6, ratio)
    q_sq = (1.0 - (0.5 / safe_ratio) ** 2).astype(complex)
    q = np.sqrt(q_sq)
    r_te = (1.0 - q) / (1.0 + q)
    phase_te = np.exp(1j * k0_t * q)
    overlap = 8.0 / (np.pi**2)
    tau_ez_mode = overlap * np.abs(
        (1.0 - r_te**2) * phase_te / (1.0 - (r_te**2) * (phase_te**2))
    ) ** 2
    # Above multi-mode onset (a/lambda > 0.75), approach high-frequency Kirchhoff limit 1
    blend_ez = np.clip((ratio - 0.55) / 0.55, 0.0, 1.0)
    tau_ez = (1.0 - blend_ez) * tau_ez_mode + blend_ez * (1.0 - 0.08 / np.maximum(ratio, 0.5))

    return {
        "tau_Ez_model": np.real(tau_ez),
        "tau_Hz_model": np.real(tau_hz),
    }


def build_single_slit_grid(
    slit_width: float,
    wavelength: float = DEFAULT_WAVELENGTH,
    screen_thickness: float = 0.15 * DEFAULT_WAVELENGTH,
    x_half_span: float | None = None,
    y_upstream: float = 1.4 * DEFAULT_WAVELENGTH,
    y_downstream: float = 4.6 * DEFAULT_WAVELENGTH,
    cpw: int = 24,
    min_slit_cells: int = 8,
    min_screen_cells: int = 4,
    n_pml: int = 10,
) -> dict[str, np.ndarray | float | int]:
    """Build a 2D graded FIT mesh for a single slit in a PEC screen.

    Enforces the Chapter 3 discretization rules:
    - background resolution ``dx_base = wavelength / cpw`` with ``cpw >= 20`` (except in
      deliberate coarse-grid convergence points),
    - at least ``min_slit_cells >= 4`` across the slit opening ``[-a/2, +a/2]``,
    - at least ``min_screen_cells >= 4`` across the screen thickness ``[-t/2, +t/2]``
      whenever ``screen_thickness > 0``.

    Parameters
    ----------
    slit_width : float
        Slit opening width a [m].
    wavelength : float, optional
        Free-space wavelength lambda [m].
    screen_thickness : float, optional
        PEC screen thickness t along y [m]. Set to 0.0 for a zero-thickness sheet.
    x_half_span : float or None, optional
        Half-width of the physical interior domain in x [m]. Defaults to
        ``max(5.5 * wavelength, 2.2 * slit_width)``.
    y_upstream : float, optional
        Distance from the screen center y = 0 to the upstream physical boundary [m].
    y_downstream : float, optional
        Distance from the screen center y = 0 to the downstream physical boundary [m].
    cpw : int, optional
        Background cells per wavelength [-].
    min_slit_cells : int, optional
        Minimum number of cells across the slit opening a [-].
    min_screen_cells : int, optional
        Minimum number of cells across the screen thickness t when t > 0 [-].
    n_pml : int, optional
        Number of CPML cells on each x and y boundary [-].

    Returns
    -------
    dict
        Grid dictionary containing ``x``, ``y``, ``z``, ``lz``, ``dx_base``,
        ``dx_slit``, ``dy_screen``, ``n_pml``, ``cells_across_slit``, and
        ``cells_across_screen``.
    """
    if x_half_span is None:
        x_half_span = max(5.5 * wavelength, 2.2 * slit_width)

    dx_base = wavelength / float(cpw)
    dx_slit = min(dx_base, slit_width / float(max(min_slit_cells, 4)))
    if screen_thickness > 0.0:
        dy_screen = min(dx_base, screen_thickness / float(max(min_screen_cells, 4)))
    else:
        dy_screen = dx_base

    x_core = graded_axis(
        [-x_half_span, -0.5 * slit_width, 0.5 * slit_width, x_half_span],
        [dx_base, dx_slit, dx_base],
        min_cells=[4, max(min_slit_cells, 4), 4],
    )
    x_core = 0.5 * (x_core - x_core[::-1])

    half_t = 0.5 * screen_thickness
    if screen_thickness > 0.0:
        y_core = graded_axis(
            [-y_upstream, -half_t, half_t, y_downstream],
            [dx_base, dy_screen, dx_base],
            min_cells=[4, max(min_screen_cells, 4), 4],
        )
    else:
        y_core = graded_axis(
            [-y_upstream, 0.0, y_downstream],
            [dx_base, dx_base],
            min_cells=[4, 4],
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

    slit_mask = (x_core >= -0.5 * slit_width - 1e-15) & (x_core <= 0.5 * slit_width + 1e-15)
    cells_across_slit = int(max(np.count_nonzero(slit_mask) - 1, 0))
    if screen_thickness > 0.0:
        scr_mask = (y_core >= -half_t - 1e-15) & (y_core <= half_t + 1e-15)
        cells_across_screen = int(max(np.count_nonzero(scr_mask) - 1, 0))
    else:
        cells_across_screen = 0

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
        "cells_across_slit": cells_across_slit,
        "cells_across_screen": cells_across_screen,
    }


def _compute_2d_farfield_from_n2f(
    n2f: Near2FarMonitor,
    theta: np.ndarray,
    wavelength: float,
    lz: float,
    ifreq: int = 0,
) -> np.ndarray:
    """Convert Wakis 3D Near2FarMonitor radiation intensity to 2D cylindrical intensity.

    In 2D (periodic over thickness L_z along z), stationary-phase integration over z
    relates the 2D angular radiation intensity U_2D(theta) [W*s^2 / m / rad] to the
    3D radiation intensity U_3D(theta_sph=pi/2, phi_sph) [W*s^2 / sr] via:
        U_2D(theta) = (2 * pi / (k_0 * L_z^2)) * U_3D(pi/2, pi/2 - theta).
    By the 2D Parseval identity, integral_{-pi/2}^{pi/2} U_2D(theta) d(theta) equals
    the time-integrated forward power per unit z-length P'_n2f = n2f.get_flux() / L_z.
    """
    k0 = 2.0 * np.pi / wavelength
    theta_arr = np.asarray(theta, dtype=float)
    theta_sph = np.full_like(theta_arr, 0.5 * np.pi)
    phi_sph = 0.5 * np.pi - theta_arr
    u3d = np.asarray(n2f.radiation_intensity(theta_sph, phi_sph)[ifreq], dtype=float)
    return (2.0 * np.pi / (k0 * (lz**2))) * u3d


def run_single_slit_simulation(
    slit_width: float,
    wavelength: float = DEFAULT_WAVELENGTH,
    screen_thickness: float = 0.15 * DEFAULT_WAVELENGTH,
    polarization: str = "Ez",
    with_screen: bool = True,
    cpw: int = 24,
    min_slit_cells: int = 8,
    min_screen_cells: int = 4,
    x_half_span: float | None = None,
    y_upstream: float = 1.4 * DEFAULT_WAVELENGTH,
    y_downstream: float = 4.6 * DEFAULT_WAVELENGTH,
    fresnel_distances: tuple[float, ...] = (),
    record_2d_dft: bool = False,
    truncate_source_at_pml: bool = False,
    theta_deg: np.ndarray | None = None,
    n_cycles: float = 13.0,
) -> dict:
    """Run a 2D FIT single-slit or empty-domain normalization simulation.

    Parameters
    ----------
    slit_width : float
        Slit opening width a [m].
    wavelength : float, optional
        Free-space wavelength lambda [m].
    screen_thickness : float, optional
        PEC screen thickness t [m]. If ``0.0``, uses a zero-thickness conformal PEC
        sheet via ``wakis.conformal.rectangle_sheet``; if ``> 0``, uses finite-thickness
        ``Box(..., "pec")`` blocks with at least ``min_screen_cells`` cells across t.
    polarization : {"Ez", "Hz"}, optional
        ``"Ez"`` for E || slit edges (s-polarization / TM_z, J_z source) or ``"Hz"``
        for E perp slit edges (p-polarization / TE_z, J_x source).
    with_screen : bool, optional
        If ``True``, places the slotted PEC screen at y = 0; if ``False``, runs the
        empty-domain incident-wave normalization on the identical mesh.
    cpw : int, optional
        Background cells per wavelength [-].
    min_slit_cells : int, optional
        Minimum cells across the slit opening a [-].
    min_screen_cells : int, optional
        Minimum cells across the screen thickness t when t > 0 [-].
    x_half_span : float or None, optional
        Half-width of the interior physical x domain [m].
    y_upstream, y_downstream : float, optional
        Upstream and downstream physical domain extents from y = 0 [m].
    fresnel_distances : tuple of float, optional
        Downstream distances L from the screen exit face y = +t/2 at which to place 1D
        transverse DFT line monitors [m].
    record_2d_dft : bool, optional
        If ``True``, records a 2D (x, y) DFT slice at f0 for field visualization.
    truncate_source_at_pml : bool, optional
        If ``False`` (recommended), the plane-wave source sheet extends all the way
        through the lateral x-CPMLs. If ``True``, truncates the source inside the
        lateral CPMLs to demonstrate edge-diffraction contamination.
    theta_deg : ndarray or None, optional
        Angular grid in degrees for far-field evaluation [-85, 85] [deg].
    n_cycles : float, optional
        Total simulated optical cycles after source injection [-].

    Returns
    -------
    dict
        Simulation results including grid metadata, fluxes, near-to-far angular
        intensities, and any requested line or 2D DFT profiles.
    """
    f0 = c0 / wavelength
    grid_info = build_single_slit_grid(
        slit_width=slit_width,
        wavelength=wavelength,
        screen_thickness=screen_thickness,
        x_half_span=x_half_span,
        y_upstream=y_upstream,
        y_downstream=y_downstream,
        cpw=cpw,
        min_slit_cells=min_slit_cells,
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

    sheets = None
    boxes = []
    left_edge = float(x[0] - 2.0 * wavelength)
    right_edge = float(x[-1] + 2.0 * wavelength)
    w_left = (-0.5 * slit_width) - left_edge
    xc_left = 0.5 * (left_edge - 0.5 * slit_width)
    w_right = right_edge - (0.5 * slit_width)
    xc_right = 0.5 * (0.5 * slit_width + right_edge)

    if with_screen:
        if screen_thickness == 0.0:
            sheets = {
                "left": rectangle_sheet((xc_left, 0.0, 0.0), (w_left, 0.0, 2.0 * lz)),
                "right": rectangle_sheet((xc_right, 0.0, 0.0), (w_right, 0.0, 2.0 * lz)),
            }
        else:
            boxes = [
                Box((xc_left, 0.0, 0.0), (w_left, screen_thickness, 2.0 * lz), "pec"),
                Box((xc_right, 0.0, 0.0), (w_right, screen_thickness, 2.0 * lz), "pec"),
            ]

    t0 = time.perf_counter()
    grid = GridFIT3D(
        x=x,
        y=y,
        z=z,
        sheets=sheets,
        use_mpi=False,
        verbose=0,
    )
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
    if truncate_source_at_pml:
        src_span_x = float((x_core[-1] - x_core[0]) - 0.8 * wavelength)
    else:
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
    w_ref = float((x_core[-1] - x_core[0]) - 0.6 * wavelength)
    y_n2f = float(half_t + 2.5 * grid_info["dy_screen"])

    monitors = []

    # Aperture / reference plane flux monitor
    w_slit_mon = float(slit_width + 2.0 * grid_info["dx_slit"])
    flux_slit = FluxMonitor(
        frequencies=freqs,
        center=(0.0, y_n2f, 0.0),
        size=(w_slit_mon, 0.0, 2.0 * lz),
    )
    monitors.append(flux_slit)

    # Full interior width flux monitor and Near2Far open surface
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

    # Transverse line DFT monitors at requested Fresnel distances
    line_monitors: dict[float, DFTMonitor] = {}
    for dist in fresnel_distances:
        y_line = float(half_t + dist)
        mon = DFTMonitor(
            frequencies=freqs,
            components=(dft_component,),
            center=(0.0, y_line, 0.0),
            size=(w_ref, 0.0, 0.0),
        )
        line_monitors[float(dist)] = mon
        monitors.append(mon)

    # Also record a transverse line at y = 0 for incident-wave flatness check
    mon_y0 = DFTMonitor(
        frequencies=freqs,
        components=(dft_component,),
        center=(0.0, 0.0, 0.0),
        size=(w_ref, 0.0, 0.0),
    )
    monitors.append(mon_y0)

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
        theta_deg = np.linspace(-84.0, 84.0, 337)
    theta_rad = np.deg2rad(theta_deg)

    # Normalize spectra by the analytic pulse Fourier transform S(f0)
    s_f0 = complex(pulse.fourier_transform(f0))
    power_norm = float(2.0 * (abs(s_f0) ** 2))

    u2d = _compute_2d_farfield_from_n2f(n2f, theta_rad, wavelength, lz, ifreq=0) / power_norm
    u2d_integral = float(np.trapezoid(u2d, theta_rad))

    p_slit_per_z = float(flux_slit.get_flux()[0] / (lz * power_norm))
    p_wide_per_z = float(flux_wide.get_flux()[0] / (lz * power_norm))
    p_n2f_per_z = float(n2f.get_flux()[0] / (lz * power_norm))

    fld_y0, coords_y0 = mon_y0.get_field(dft_component)
    field_y0 = np.squeeze(fld_y0[0]) / s_f0
    x_y0 = coords_y0[0]

    line_profiles = {}
    for dist, mon in line_monitors.items():
        fld_d, coords_d = mon.get_field(dft_component)
        line_profiles[dist] = {
            "x": coords_d[0].copy(),
            "field": (np.squeeze(fld_d[0]) / s_f0).copy(),
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
        "wavelength": float(wavelength),
        "screen_thickness": float(screen_thickness),
        "polarization": pol,
        "with_screen": bool(with_screen),
        "cpw": int(cpw),
        "nx": int(grid.Nx),
        "ny": int(grid.Ny),
        "nz": int(grid.Nz),
        "n_cells": int(grid.Nx * grid.Ny * grid.Nz),
        "n_steps": int(n_steps),
        "dt": float(solver.dt),
        "elapsed_s": float(elapsed_s),
        "cells_across_slit": int(grid_info["cells_across_slit"]),
        "cells_across_screen": int(grid_info["cells_across_screen"]),
        "w_ref": w_ref,
        "w_slit_mon": w_slit_mon,
        "p_slit_per_z": p_slit_per_z,
        "p_wide_per_z": p_wide_per_z,
        "p_n2f_per_z": p_n2f_per_z,
        "u2d_integral": u2d_integral,
        "parseval_ratio": u2d_integral / max(abs(p_n2f_per_z), 1e-30),
        "theta_deg": theta_deg.copy(),
        "theta_rad": theta_rad.copy(),
        "u2d": u2d.copy(),
        "x_y0": x_y0.copy(),
        "field_y0": field_y0.copy(),
        "line_profiles": line_profiles,
        "map_2d": map_2d,
    }


def run_preflight_summary(
    wavelength: float = DEFAULT_WAVELENGTH,
    slit_width: float = 2.0 * DEFAULT_WAVELENGTH,
    screen_thickness: float = 0.15 * DEFAULT_WAVELENGTH,
    cpw: int = 24,
) -> dict:
    """Run ``wakis.estimate.estimate`` and ``preflight`` on the representative 2D slit grid."""
    grid_info = build_single_slit_grid(
        slit_width=slit_width,
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
    solver_mb = float(est.breakdown.get("solver (host)", 0.0) / 1e6)
    mon_mb = float(est.breakdown.get("monitors", 0.0) / 1e6)
    return {
        "n_cells": int(est.cells),
        "solver_ram_mb": solver_mb,
        "monitor_ram_mb": mon_mb,
        "total_ram_mb": float(est.host_bytes / 1e6),
        "wall_clock_s": float(est.seconds),
        "dt": dt_est,
        "cells_across_slit": int(grid_info["cells_across_slit"]),
        "cells_across_screen": int(grid_info["cells_across_screen"]),
        "ok": len(problems) == 0,
        "warnings": list(problems),
    }


def run_source_benchmark_task(quick: bool = False) -> dict:
    """Compare full-width plane-wave source (through CPML) vs PML-truncated source."""
    wavelength = DEFAULT_WAVELENGTH
    cpw = 20 if quick else 24
    slit_width = 2.0 * wavelength

    full_src = run_single_slit_simulation(
        slit_width=slit_width,
        wavelength=wavelength,
        with_screen=False,
        cpw=cpw,
        truncate_source_at_pml=False,
        x_half_span=4.5 * wavelength,
        y_upstream=1.2 * wavelength,
        y_downstream=2.2 * wavelength,
        n_cycles=11.0 if quick else 13.0,
    )
    trunc_src = run_single_slit_simulation(
        slit_width=slit_width,
        wavelength=wavelength,
        with_screen=False,
        cpw=cpw,
        truncate_source_at_pml=True,
        x_half_span=4.5 * wavelength,
        y_upstream=1.2 * wavelength,
        y_downstream=2.2 * wavelength,
        n_cycles=11.0 if quick else 13.0,
    )

    x_prof = full_src["x_y0"]
    central_mask = np.abs(x_prof) <= 2.5 * wavelength
    amp_full = np.abs(full_src["field_y0"])
    amp_trunc = np.abs(trunc_src["field_y0"])
    mean_full = float(np.mean(amp_full[central_mask]))
    ripple_full = float(
        (np.max(amp_full[central_mask]) - np.min(amp_full[central_mask])) / mean_full
    )
    ripple_trunc = float(
        (np.max(amp_trunc[central_mask]) - np.min(amp_trunc[central_mask]))
        / float(np.mean(amp_trunc[central_mask]))
    )

    # Compare time-averaged incident intensity from FluxMonitor vs |E_z|^2 / (2 * Z_0)
    eta0 = 376.730313668
    i_inc_flux = full_src["p_wide_per_z"] / full_src["w_ref"]
    i_inc_field = 0.5 * (mean_full**2) / eta0
    flux_field_rel_diff = abs(i_inc_flux - i_inc_field) / i_inc_field

    return {
        "wavelength": np.array(wavelength),
        "x_um": x_prof * 1e6,
        "norm_amp_full": amp_full / mean_full,
        "norm_amp_trunc": amp_trunc / mean_full,
        "phase_full_rad": np.angle(full_src["field_y0"] / np.mean(full_src["field_y0"][central_mask])),
        "phase_trunc_rad": np.angle(
            trunc_src["field_y0"] / np.mean(full_src["field_y0"][central_mask])
        ),
        "ripple_full": np.array(ripple_full),
        "ripple_trunc": np.array(ripple_trunc),
        "flux_field_rel_diff": np.array(flux_field_rel_diff),
        "parseval_ratio_empty": np.array(full_src["parseval_ratio"]),
    }


def run_fresnel_task(quick: bool = False) -> dict:
    """Compute near-field Fresnel line profiles across N_F in {1.5, 1.0, 0.5, 0.25}."""
    wavelength = DEFAULT_WAVELENGTH
    slit_width = 2.8 * wavelength  # a = 2.8 * lambda
    screen_thickness = 0.0  # Knife-edge conformal PEC sheet to isolate pure aperture diffraction
    cpw = 20 if quick else 22

    # N_F = a^2 / (4 * lambda * L) -> L / lambda = (1.31, 1.96, 3.92, 7.84)
    nf_targets = np.array([1.5, 1.0, 0.5, 0.25], dtype=float)
    distances = tuple(float((slit_width**2) / (4.0 * wavelength * nf)) for nf in nf_targets)

    common_kwargs = dict(
        slit_width=slit_width,
        wavelength=wavelength,
        screen_thickness=screen_thickness,
        polarization="Ez",
        cpw=cpw,
        min_slit_cells=12,
        min_screen_cells=4,
        x_half_span=5.5 * wavelength,
        y_upstream=1.2 * wavelength,
        y_downstream=8.4 * wavelength,
        fresnel_distances=distances,
        n_cycles=12.0 if quick else 14.0,
    )

    inc = run_single_slit_simulation(with_screen=False, record_2d_dft=False, **common_kwargs)
    scat = run_single_slit_simulation(with_screen=True, record_2d_dft=True, **common_kwargs)

    # Normalize each detector line by the empty-domain incident intensity on the same line
    x_line = scat["line_profiles"][distances[0]]["x"]
    fit_profiles = []
    cornu_profiles = []
    rs1_profiles = []
    l2_errors_cornu = []
    l2_errors_rs1 = []

    roi_mask = np.abs(x_line) <= 2.4 * slit_width
    for dist in distances:
        e_scat = scat["line_profiles"][dist]["field"]
        e_inc = inc["line_profiles"][dist]["field"]
        i_inc_ref = float(np.mean(np.abs(e_inc[ np.abs(x_line) <= 1.5 * slit_width ]) ** 2))
        i_fit = (np.abs(e_scat) ** 2) / i_inc_ref
        i_cornu = fresnel_cornu_profile(x_line, slit_width, wavelength, dist)
        i_rs1 = rayleigh_sommerfeld_line_profile(
            x_line, slit_width, wavelength, dist, kind="I"
        )

        err_cornu = float(
            np.linalg.norm(i_fit[roi_mask] - i_cornu[roi_mask])
            / np.linalg.norm(i_cornu[roi_mask])
        )
        err_rs1 = float(
            np.linalg.norm(i_fit[roi_mask] - i_rs1[roi_mask])
            / np.linalg.norm(i_rs1[roi_mask])
        )
        fit_profiles.append(i_fit)
        cornu_profiles.append(i_cornu)
        rs1_profiles.append(i_rs1)
        l2_errors_cornu.append(err_cornu)
        l2_errors_rs1.append(err_rs1)

    e_inc_y0 = float(np.mean(np.abs(inc["field_y0"][np.abs(inc["x_y0"]) <= 1.5 * slit_width])))
    map_x = scat["map_2d"]["x"]
    map_y = scat["map_2d"]["y"]
    map_ez_real = np.real(scat["map_2d"]["field"]) / e_inc_y0
    map_intensity = (np.abs(scat["map_2d"]["field"]) / e_inc_y0) ** 2

    return {
        "wavelength": np.array(wavelength),
        "slit_width": np.array(slit_width),
        "screen_thickness": np.array(screen_thickness),
        "nf_values": nf_targets,
        "distances_over_lam": np.array(distances) / wavelength,
        "x_over_a": x_line / slit_width,
        "fit_profiles": np.asarray(fit_profiles),
        "cornu_profiles": np.asarray(cornu_profiles),
        "rs1_profiles": np.asarray(rs1_profiles),
        "l2_errors_cornu": np.asarray(l2_errors_cornu),
        "l2_errors_rs1": np.asarray(l2_errors_rs1),
        "map_x_over_lam": map_x / wavelength,
        "map_y_over_lam": map_y / wavelength,
        "map_ez_real": map_ez_real,
        "map_intensity": map_intensity,
    }


def run_farfield_task(quick: bool = False) -> dict:
    """Compare 2D Near2Far angular patterns with Fraunhofer, Kirchhoff, and RS-I/II."""
    wavelength = DEFAULT_WAVELENGTH
    cpw = 20 if quick else 24
    screen_thickness = 0.12 * wavelength
    theta_deg = np.linspace(-82.0, 82.0, 329)
    theta_rad = np.deg2rad(theta_deg)

    a_over_lam_list = [4.0, 2.0, 1.0, 0.6]
    u2d_ez_norm = []
    u2d_hz_norm = []
    fraunhofer_list = []
    kirchhoff_list = []
    rs1_list = []
    rs2_list = []
    l2_ez_vs_fraun = []
    l2_ez_vs_rs1 = []
    l2_hz_vs_rs2 = []
    parseval_ez = []
    parseval_hz = []

    ang_mask = np.abs(theta_deg) <= 72.0

    for ratio in a_over_lam_list:
        a = ratio * wavelength
        x_span = max(6.0 * wavelength, 2.4 * a)
        sim_ez = run_single_slit_simulation(
            slit_width=a,
            wavelength=wavelength,
            screen_thickness=screen_thickness,
            polarization="Ez",
            with_screen=True,
            cpw=cpw,
            min_slit_cells=8,
            min_screen_cells=4,
            x_half_span=x_span,
            y_upstream=1.2 * wavelength,
            y_downstream=2.2 * wavelength,
            theta_deg=theta_deg,
            n_cycles=12.0 if quick else 14.0,
        )
        sim_hz = run_single_slit_simulation(
            slit_width=a,
            wavelength=wavelength,
            screen_thickness=screen_thickness,
            polarization="Hz",
            with_screen=True,
            cpw=cpw,
            min_slit_cells=8,
            min_screen_cells=4,
            x_half_span=x_span,
            y_upstream=1.2 * wavelength,
            y_downstream=2.2 * wavelength,
            theta_deg=theta_deg,
            n_cycles=12.0 if quick else 14.0,
        )

        idx0 = int(np.argmin(np.abs(theta_deg)))
        p_ez = sim_ez["u2d"] / max(float(sim_ez["u2d"][idx0]), 1e-30)
        p_hz = sim_hz["u2d"] / max(float(sim_hz["u2d"][idx0]), 1e-30)

        p_fraun = fraunhofer_pattern(theta_rad, a, wavelength)
        p_kirch = kirchhoff_obliquity_pattern(theta_rad, a, wavelength)
        p_rs1 = rayleigh_sommerfeld_farfield(theta_rad, a, wavelength, kind="I")
        p_rs2 = rayleigh_sommerfeld_farfield(theta_rad, a, wavelength, kind="II")

        u2d_ez_norm.append(p_ez)
        u2d_hz_norm.append(p_hz)
        fraunhofer_list.append(p_fraun)
        kirchhoff_list.append(p_kirch)
        rs1_list.append(p_rs1)
        rs2_list.append(p_rs2)

        l2_ez_vs_fraun.append(
            float(
                np.linalg.norm(p_ez[ang_mask] - p_fraun[ang_mask])
                / np.linalg.norm(p_fraun[ang_mask])
            )
        )
        l2_ez_vs_rs1.append(
            float(
                np.linalg.norm(p_ez[ang_mask] - p_rs1[ang_mask])
                / np.linalg.norm(p_rs1[ang_mask])
            )
        )
        l2_hz_vs_rs2.append(
            float(
                np.linalg.norm(p_hz[ang_mask] - p_rs2[ang_mask])
                / np.linalg.norm(p_rs2[ang_mask])
            )
        )
        parseval_ez.append(sim_ez["parseval_ratio"])
        parseval_hz.append(sim_hz["parseval_ratio"])

    return {
        "a_over_lam": np.asarray(a_over_lam_list, dtype=float),
        "theta_deg": theta_deg,
        "u2d_ez_norm": np.asarray(u2d_ez_norm),
        "u2d_hz_norm": np.asarray(u2d_hz_norm),
        "fraunhofer": np.asarray(fraunhofer_list),
        "kirchhoff": np.asarray(kirchhoff_list),
        "rs1": np.asarray(rs1_list),
        "rs2": np.asarray(rs2_list),
        "l2_ez_vs_fraun": np.asarray(l2_ez_vs_fraun),
        "l2_ez_vs_rs1": np.asarray(l2_ez_vs_rs1),
        "l2_hz_vs_rs2": np.asarray(l2_hz_vs_rs2),
        "parseval_ez": np.asarray(parseval_ez),
        "parseval_hz": np.asarray(parseval_hz),
    }


def run_transmission_task(quick: bool = False) -> dict:
    """Compute normalized transmitted power tau(a/lambda) across polarization & thickness."""
    wavelength = DEFAULT_WAVELENGTH
    cpw = 20 if quick else 22
    if quick:
        ratios = np.array([0.32, 0.45, 0.65, 1.15, 2.00], dtype=float)
    else:
        ratios = np.array([0.30, 0.40, 0.50, 0.65, 0.90, 1.40, 2.00], dtype=float)

    t_thick = 0.20 * wavelength
    x_span = 4.2 * wavelength if quick else 4.5 * wavelength

    # Normalization runs (empty domain) for E_z and H_z
    inc_ez = run_single_slit_simulation(
        slit_width=1.0 * wavelength,
        wavelength=wavelength,
        screen_thickness=t_thick,
        polarization="Ez",
        with_screen=False,
        cpw=cpw,
        x_half_span=x_span,
        y_upstream=1.2 * wavelength,
        y_downstream=2.0 * wavelength,
        n_cycles=11.0 if quick else 13.0,
    )
    inc_hz = run_single_slit_simulation(
        slit_width=1.0 * wavelength,
        wavelength=wavelength,
        screen_thickness=t_thick,
        polarization="Hz",
        with_screen=False,
        cpw=cpw,
        x_half_span=x_span,
        y_upstream=1.2 * wavelength,
        y_downstream=2.0 * wavelength,
        n_cycles=11.0 if quick else 13.0,
    )
    i_inc_ez = inc_ez["p_wide_per_z"] / inc_ez["w_ref"]
    i_inc_hz = inc_hz["p_wide_per_z"] / inc_hz["w_ref"]

    tau_ez_thick = []
    tau_hz_thick = []
    tau_ez_sheet = []
    tau_hz_sheet = []

    for ratio in ratios:
        a = float(ratio * wavelength)
        # Finite-thickness PEC screen (t = 0.20 * lambda)
        res_ez_t = run_single_slit_simulation(
            slit_width=a,
            wavelength=wavelength,
            screen_thickness=t_thick,
            polarization="Ez",
            with_screen=True,
            cpw=cpw,
            min_slit_cells=6,
            min_screen_cells=4,
            x_half_span=x_span,
            y_upstream=1.2 * wavelength,
            y_downstream=2.0 * wavelength,
            n_cycles=11.0 if quick else 13.0,
        )
        res_hz_t = run_single_slit_simulation(
            slit_width=a,
            wavelength=wavelength,
            screen_thickness=t_thick,
            polarization="Hz",
            with_screen=True,
            cpw=cpw,
            min_slit_cells=6,
            min_screen_cells=4,
            x_half_span=x_span,
            y_upstream=1.2 * wavelength,
            y_downstream=2.0 * wavelength,
            n_cycles=11.0 if quick else 13.0,
        )
        tau_ez_thick.append(res_ez_t["p_n2f_per_z"] / (a * i_inc_ez))
        tau_hz_thick.append(res_hz_t["p_n2f_per_z"] / (a * i_inc_hz))

        # Zero-thickness conformal PEC sheet (t = 0)
        res_ez_s = run_single_slit_simulation(
            slit_width=a,
            wavelength=wavelength,
            screen_thickness=0.0,
            polarization="Ez",
            with_screen=True,
            cpw=cpw,
            min_slit_cells=6,
            x_half_span=x_span,
            y_upstream=1.2 * wavelength,
            y_downstream=2.0 * wavelength,
            n_cycles=11.0 if quick else 13.0,
        )
        res_hz_s = run_single_slit_simulation(
            slit_width=a,
            wavelength=wavelength,
            screen_thickness=0.0,
            polarization="Hz",
            with_screen=True,
            cpw=cpw,
            min_slit_cells=6,
            x_half_span=x_span,
            y_upstream=1.2 * wavelength,
            y_downstream=2.0 * wavelength,
            n_cycles=11.0 if quick else 13.0,
        )
        tau_ez_sheet.append(res_ez_s["p_n2f_per_z"] / (a * i_inc_ez))
        tau_hz_sheet.append(res_hz_s["p_n2f_per_z"] / (a * i_inc_hz))

    fine_ratios = np.linspace(0.25, 2.05, 181)
    models = fabry_perot_slit_transmission_models(fine_ratios, t_over_lam=0.20)

    return {
        "a_over_lam": ratios,
        "t_over_lam": np.array(0.20),
        "tau_ez_thick": np.asarray(tau_ez_thick),
        "tau_hz_thick": np.asarray(tau_hz_thick),
        "tau_ez_sheet": np.asarray(tau_ez_sheet),
        "tau_hz_sheet": np.asarray(tau_hz_sheet),
        "model_a_over_lam": fine_ratios,
        "model_tau_ez_thick": models["tau_Ez_model"],
        "model_tau_hz_thick": models["tau_Hz_model"],
    }


def run_resolution_task(quick: bool = False) -> dict:
    """Grid convergence study over cells-per-wavelength (cpw) for a = 2 * lambda."""
    wavelength = DEFAULT_WAVELENGTH
    slit_width = 2.0 * wavelength
    screen_thickness = 0.15 * wavelength
    cpw_list = [14, 20, 28] if quick else [12, 16, 20, 28, 36]

    theta_deg = np.linspace(-80.0, 80.0, 321)
    theta_rad = np.deg2rad(theta_deg)
    rs1_ref = rayleigh_sommerfeld_farfield(theta_rad, slit_width, wavelength, kind="I")
    ang_mask = np.abs(theta_deg) <= 65.0

    l2_vs_rs1 = []
    tau_values = []
    null_deg_values = []
    cells_slit_list = []
    cells_screen_list = []
    elapsed_list = []
    patterns = []

    for cpw in cpw_list:
        # Let mesh scale naturally with cpw so we observe genuine convergence
        slit_cells = max(4, int(round(2.0 * cpw)))
        scr_cells = max(2, int(round(0.15 * cpw)))
        inc = run_single_slit_simulation(
            slit_width=slit_width,
            wavelength=wavelength,
            screen_thickness=screen_thickness,
            polarization="Ez",
            with_screen=False,
            cpw=cpw,
            min_slit_cells=slit_cells,
            min_screen_cells=scr_cells,
            x_half_span=4.8 * wavelength,
            y_upstream=1.2 * wavelength,
            y_downstream=2.0 * wavelength,
            theta_deg=theta_deg,
            n_cycles=12.0,
        )
        scat = run_single_slit_simulation(
            slit_width=slit_width,
            wavelength=wavelength,
            screen_thickness=screen_thickness,
            polarization="Ez",
            with_screen=True,
            cpw=cpw,
            min_slit_cells=slit_cells,
            min_screen_cells=scr_cells,
            x_half_span=4.8 * wavelength,
            y_upstream=1.2 * wavelength,
            y_downstream=2.0 * wavelength,
            theta_deg=theta_deg,
            n_cycles=12.0,
        )
        idx0 = int(np.argmin(np.abs(theta_deg)))
        pat = scat["u2d"] / max(float(scat["u2d"][idx0]), 1e-30)
        err = float(
            np.linalg.norm(pat[ang_mask] - rs1_ref[ang_mask]) / np.linalg.norm(rs1_ref[ang_mask])
        )
        i_inc = inc["p_wide_per_z"] / inc["w_ref"]
        tau = scat["p_n2f_per_z"] / (slit_width * i_inc)

        # Locate first diffraction null near arcsin(lambda / a) = 30 deg
        null_win = (theta_deg >= 20.0) & (theta_deg <= 42.0)
        theta_win = theta_deg[null_win]
        pat_win = pat[null_win]
        imin = int(np.argmin(pat_win))
        null_deg = float(theta_win[imin])

        l2_vs_rs1.append(err)
        tau_values.append(tau)
        null_deg_values.append(null_deg)
        cells_slit_list.append(scat["cells_across_slit"])
        cells_screen_list.append(scat["cells_across_screen"])
        elapsed_list.append(scat["elapsed_s"])
        patterns.append(pat)

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
        "rs1_ref": rs1_ref,
        "l2_vs_rs1": np.asarray(l2_vs_rs1),
        "rel_to_finest": np.asarray(rel_to_finest),
        "tau_values": np.asarray(tau_values),
        "null_deg_values": np.asarray(null_deg_values),
        "cells_across_slit": np.asarray(cells_slit_list, dtype=int),
        "cells_across_screen": np.asarray(cells_screen_list, dtype=int),
        "elapsed_s": np.asarray(elapsed_list),
    }


def _build_schematic_solver(
    slit_width: float,
    wavelength: float = DEFAULT_WAVELENGTH,
    screen_thickness: float = 0.15 * DEFAULT_WAVELENGTH,
    cpw: int = 24,
    y_upstream: float = 1.4 * DEFAULT_WAVELENGTH,
    y_downstream: float = 3.2 * DEFAULT_WAVELENGTH,
) -> tuple[SolverFIT3D, dict, list[Box], CurrentSource, list]:
    """Build a lightweight 2D single-slit solver, structures, source, and monitors for schematics/overlays."""
    f0 = c0 / wavelength
    grid_info = build_single_slit_grid(
        slit_width=slit_width,
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
    w_left = (-0.5 * slit_width) - left_end
    xc_left = 0.5 * (left_end - 0.5 * slit_width)
    w_right = right_end - 0.5 * slit_width
    xc_right = 0.5 * (0.5 * slit_width + right_end)
    boxes = [
        Box((xc_left, 0.0, 0.0), (w_left, screen_thickness, 2.0 * lz), "pec"),
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
        component="z",
        center=(0.0, y_src, 0.0),
        size=(src_span_x, 0.0, 2.0 * lz),
        amplitude=1.0,
    )

    freqs = np.array([f0])
    w_ref = float((x_core[-1] - x_core[0]) - 0.5 * wavelength)
    y_n2f = float(half_t + 2.5 * grid_info["dy_screen"])
    flux_slit = FluxMonitor(
        frequencies=freqs,
        center=(0.0, 0.0, 0.0),
        size=(slit_width, 0.0, 2.0 * lz),
    )
    n2f = Near2FarMonitor(
        frequencies=freqs,
        center=(0.0, y_n2f, 0.0),
        size=(w_ref, 0.0, 2.0 * lz),
    )
    y_nf = float(half_t + (slit_width**2) / (4.0 * wavelength * 0.5))
    mon_nf = DFTMonitor(
        frequencies=freqs,
        components=("Ez",),
        center=(0.0, y_nf, 0.0),
        size=(w_ref, 0.0, 2.0 * lz),
    )
    return solver, grid_info, boxes, source, [flux_slit, n2f, mon_nf]


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
    for txt in ax.texts:
        s = txt.get_text()
        if "$-y$ boundary:" in s or "-y boundary:" in s:
            txt.set_text(
                r"$\pm y$ boundaries: CPML (10 cells, $0.25\;\mu\mathrm{m}$) $\mid$ Out-of-plane ($\pm z$: Periodic)"
            )
        elif "PEC" in s:
            txt.set_text(r"PEC ($\sigma \to \infty$)")
            txt.set_position((x_left_jaw if pec_count == 0 else x_right_jaw, -0.15))
            txt.set_ha("center")
            txt.set_va("top")
            pec_count += 1
        elif "CurrentSource" in s:
            txt.set_position((0.0, y_src - 0.10))
            txt.set_ha("center")
            txt.set_va("top")
        elif "Near2Far" in s:
            txt.set_text("Near2FarMonitor")
            txt.set_position((0.0, y_n2f + 0.10))
            txt.set_ha("center")
            txt.set_va("bottom")
        elif "FluxMonitor" in s:
            txt.set_position((0.0, -0.15))
            txt.set_ha("center")
            txt.set_va("top")
        elif "DFTMonitor" in s:
            txt.set_position((0.0, y_dft + 0.08))
            txt.set_ha("center")
            txt.set_va("bottom")
    _fix_unit_spaces_in_texts(ax)


def plot_setup_schematic(
    output_path: str | Path,
    wavelength: float = DEFAULT_WAVELENGTH,
) -> dict:
    """Render ``00a_setup_geometry.png``, ``00b_setup_mesh_and_excitation.png``, and ``00_setup_table.md``."""
    out_p = Path(output_path)
    out_dir = out_p.parent if out_p.suffix else out_p
    slit_width = 2.0 * wavelength
    screen_thickness = 0.15 * wavelength
    cpw = 24
    solver, grid_info, boxes, source, monitors = _build_schematic_solver(
        slit_width=slit_width,
        wavelength=wavelength,
        screen_thickness=screen_thickness,
        cpw=cpw,
        y_upstream=1.4 * wavelength,
        y_downstream=4.6 * wavelength,
    )
    boxes[0].name = "PEC screen (left jaw)"
    boxes[1].name = "PEC screen (right jaw)"
    flux_slit, n2f, mon_nf = monitors
    w_ref = float(n2f.size[0])
    y_n2f = float(n2f.center[1])
    f0 = float(n2f.frequencies[0])
    n2f_plane = {
        "kind": "plane",
        "type_name": "Near2FarMonitor",
        "short_word": "N2F",
        "desc": f"near-to-far plane (Near2FarMonitor, y = {y_n2f * 1e9:.0f}\u2009nm)",
        "label": f"Near2FarMonitor (y = {y_n2f * 1e9:.0f}\u2009nm)",
        "pos_str": f"y = {y_n2f * 1e9:.0f}\u2009nm",
        "freq_str": f"f = {f0 * 1e-12:.1f}\u2009THz",
        "frequencies": n2f.frequencies,
        "h_bounds": (y_n2f, y_n2f),
        "v_bounds": (-0.5 * w_ref, 0.5 * w_ref),
    }
    return schematic.save_setup_figures(
        solver,
        out_dir,
        plane="yx",
        unit="µm",
        structures=boxes,
        sources=[source],
        monitors=[flux_slit, n2f_plane, mon_nf],
        wavelength=wavelength,
        min_feature=screen_thickness,
        title_geometry="2D single-slit diffraction setup: PEC screen, plane-wave source, monitors & CPML (YX plane)",
        title_mesh="2D single-slit diffraction setup: graded mesh zoom at slit jaws, wave packet & pulse",
        variations=[
            {
                "name": "Task 1 · Plane-wave source & Fresnel near-field sweep",
                "details": "Full-width vs PML-truncated source benchmark + Fresnel line monitors at N_F ∈ {2.0, 1.0, 0.5, 0.25} (a = 2λ, E_z)",
            },
            {
                "name": "Task 2 · Far-field angular patterns (Near2FarMonitor)",
                "details": "Slit width sweep a/λ ∈ {4.0, 2.0, 1.0, 0.6} for both E_z (TM_z, RS-I) and H_z (TE_z, RS-II) polarizations",
            },
            {
                "name": "Task 3 · Subwavelength transmission & waveguide cutoff",
                "details": "Normalized transmission τ(a/λ) for a/λ ∈ [0.30, 2.00] comparing thick PEC screen (t = 0.20λ) vs zero-thickness sheet (t = 0)",
            },
            {
                "name": "Task 4 · Mesh resolution convergence study",
                "details": "Background resolution sweep cpw ∈ {12, 16, 20, 28, 36} cells/λ for a = 2λ, t = 0.15λ",
            },
        ],
    )


def plot_source_and_fresnel(
    src_data: dict,
    fresnel_data: dict,
    output_path: str | Path,
) -> None:
    """Generate the 3-panel figure for plane-wave injection & Fresnel near-field regimes."""
    wavelength = float(fresnel_data.get("wavelength", DEFAULT_WAVELENGTH))
    a_fres = float(fresnel_data.get("slit_width", 2.8 * wavelength))
    t_fres = max(float(fresnel_data.get("screen_thickness", 0.0)), 0.12 * wavelength)
    solver_fres, _, boxes_fres, _, _ = _build_schematic_solver(
        slit_width=a_fres,
        wavelength=wavelength,
        screen_thickness=t_fres,
        cpw=22,
        y_upstream=1.2 * wavelength,
        y_downstream=8.2 * wavelength,
    )

    with figstyle.apply_style():
        fig, axes = figstyle.new_figure(nrows=1, ncols=3, width=15.0, height=4.8)

        # Panel (a): Plane-wave source transverse uniformity (full-width vs PML-truncated)
        ax = axes[0]
        ax.plot(
            src_data["x_um"],
            src_data["norm_amp_full"],
            **figstyle.series_style(0, with_marker=False, lw=2.1),
            label=f"Full-width source (into CPML): ripple = {float(src_data['ripple_full']) * 100:.2f}%",
        )
        ax.plot(
            src_data["x_um"],
            src_data["norm_amp_trunc"],
            **figstyle.series_style(1, with_marker=False, lw=1.8),
            label=f"Truncated inside CPML: ripple = {float(src_data['ripple_trunc']) * 100:.1f}%",
        )
        ax.axvspan(
            -1.5,
            1.5,
            color="#E0F2FE",
            alpha=0.55,
            label=r"Central aperture zone $|x| \leq 2.5\lambda$",
        )
        schematic.overlay_geometry(
            ax,
            solver_fres,
            plane="xy",
            unit="µm",
            orientation="1d",
            structures=boxes_fres,
            show_materials=False,
            show_boundaries=True,
        )
        ax.set_xlabel(r"Transverse position $x$ [µm]")
        ax.set_ylabel(r"Normalized incident amplitude $|E_z^{\mathrm{inc}}(x, 0)| / E_0$ [-]")
        ax.set_title("(a) Source termination at lateral CPML")
        ax.set_ylim(0.65, 1.30)
        figstyle.spectrum_grid(ax)
        ax.legend(loc="lower center", fontsize=7.8)

        # Panel (b): 2D intensity map |E_z(x, y)|^2 / |E_inc|^2 with overlay_geometry
        ax = axes[1]
        lam_um = wavelength * 1e6
        x_um = fresnel_data["map_x_over_lam"] * lam_um
        y_um = fresnel_data["map_y_over_lam"] * lam_um
        inten = fresnel_data["map_intensity"]
        im = ax.pcolormesh(
            x_um,
            y_um,
            inten.T,
            shading="auto",
            cmap=figstyle.CMAP_INTENSITY,
            vmin=0.0,
            vmax=min(float(np.percentile(inten, 99.5)), 1.8),
        )
        fig.colorbar(im, ax=ax, label=r"$|E_z(x, y)|^2 / |E_{\mathrm{inc}}|^2$ [-]")
        schematic.overlay_geometry(
            ax,
            solver_fres,
            plane="xy",
            unit="µm",
            orientation="2d",
            structures=boxes_fres,
            show_boundaries=True,
            show_materials=True,
            fill_materials=True,
            material_alpha=0.35,
            outline_color="#FFFFFF",
            label_materials=False,
        )
        ax.text(
            0.5 * (float(x_um[0]) - 0.5 * a_fres * 1e6),
            0.16,
            "PEC screen",
            color="white",
            fontsize=7.6,
            fontweight="bold",
            ha="center",
            va="bottom",
            zorder=8,
        )
        for nf_val, d_lam in zip(
            fresnel_data["nf_values"], fresnel_data["distances_over_lam"]
        ):
            y_line_um = float(d_lam) * lam_um
            ax.axhline(y_line_um, color="#009E73", ls="--", lw=1.1, zorder=7)
            y_txt = y_line_um - 0.25 if nf_val <= 0.3 else y_line_um + 0.04
            ax.text(
                float(x_um[-1]) - 0.12,
                y_txt,
                rf"$N_F={nf_val:g}$ ($L={d_lam:.2f}\lambda$)",
                color="white",
                fontsize=7.5,
                fontweight="bold",
                ha="right",
                zorder=8,
            )
        ax.set_xlim(float(x_um[0]), float(x_um[-1]))
        ax.set_ylim(float(y_um[0]), float(y_um[-1]))
        ax.set_aspect("auto")
        ax.set_xlabel(r"Transverse coordinate $x$ [µm]")
        ax.set_ylabel(r"Propagation coordinate $y$ [µm]")
        ax.set_title(r"(b) 2D FIT near field ($a = 2.8\lambda = 1.68\;\mu\mathrm{m}$)")

        # Panel (c): Fresnel line profiles across N_F in {1.5, 1.0, 0.5, 0.25}
        ax_p = axes[2]
        x_over_a = fresnel_data["x_over_a"]
        nf_vals = fresnel_data["nf_values"]
        d_lams = fresnel_data["distances_over_lam"]
        fit_p = fresnel_data["fit_profiles"]
        cornu_p = fresnel_data["cornu_profiles"]
        rs1_p = fresnel_data["rs1_profiles"]
        err_rs = fresnel_data["l2_errors_rs1"]

        offsets = [0.0, 1.25, 2.50, 3.75]
        for idx in range(len(nf_vals)):
            off = offsets[idx]
            nf = float(nf_vals[idx])
            dl = float(d_lams[idx])
            ax_p.plot(
                x_over_a,
                fit_p[idx] + off,
                color=figstyle.PALETTE[idx % len(figstyle.PALETTE)],
                lw=2.0,
                ls="-",
                label=rf"FIT $N_F={nf:g}$ ($L^2={err_rs[idx]*100:.1f}\%$)",
            )
            ax_p.plot(
                x_over_a,
                rs1_p[idx] + off,
                color=figstyle.INK,
                lw=1.3,
                ls="--",
                label="RS-I exact" if idx == 0 else "_nolegend_",
            )
            ax_p.plot(
                x_over_a,
                cornu_p[idx] + off,
                color="#D55E00",
                lw=1.2,
                ls=":",
                label="Cornu spiral" if idx == 0 else "_nolegend_",
            )
        ax_p.axvline(-0.5, color="#64748B", ls=":", lw=1.0)
        ax_p.axvline(0.5, color="#64748B", ls=":", lw=1.0, label=r"Slit edges $x/a = \pm 0.5$")
        ax_p.set_xlim(-2.2, 2.2)
        ax_p.set_ylim(-0.08, 6.25)
        ax_p.set_xlabel(r"Normalized transverse coordinate $x / a$ [-]")
        ax_p.set_ylabel(r"Normalized intensity $|E_z|^2 / |E_{\mathrm{inc}}|^2$ (offset) [-]")
        ax_p.set_title(r"(c) Fresnel profiles ($N_F = 1.5, 1.0, 0.5, 0.25$)")
        figstyle.spectrum_grid(ax_p)
        ax_p.legend(loc="upper center", ncol=2, fontsize=6.5)

        figstyle.save(fig, output_path, close=True)


def plot_farfield_comparison(
    ff_data: dict,
    output_path: str | Path,
) -> None:
    """Generate the 3-panel far-field angular pattern comparison figure."""
    theta_deg = ff_data["theta_deg"]
    ratios = ff_data["a_over_lam"]

    with figstyle.apply_style():
        fig, axes = figstyle.new_figure(nrows=1, ncols=3, width=15.0, height=4.8)

        # Panel (a): Wide slit a/lambda = 4.0 (idx 0)
        # Panel (b): Intermediate slit a/lambda = 2.0 (idx 1)
        for idx in (0, 1):
            ax = axes[idx]
            ratio = float(ratios[idx])
            p_ez = ff_data["u2d_ez_norm"][idx]
            p_hz = ff_data["u2d_hz_norm"][idx]
            p_fraun = ff_data["fraunhofer"][idx]
            p_kirch = ff_data["kirchhoff"][idx]
            p_rs1 = ff_data["rs1"][idx]

            ax.plot(
                theta_deg,
                p_ez,
                color=figstyle.PALETTE[0],
                lw=2.0,
                ls="-",
                label=r"FIT $\mathbf{E} \parallel$ slit ($E_z$, TM$_z$)",
            )
            ax.plot(
                theta_deg,
                p_hz,
                color=figstyle.PALETTE[1],
                lw=1.9,
                ls="-.",
                label=r"FIT $\mathbf{E} \perp$ slit ($H_z$, TE$_z$)",
            )
            ax.plot(
                theta_deg,
                p_fraun,
                color=figstyle.INK,
                lw=1.3,
                ls="--",
                label=r"Fraunhofer $\mathrm{sinc}^2(\beta)$ (= RS-II)",
            )
            ax.plot(
                theta_deg,
                p_rs1,
                color=figstyle.PALETTE[2],
                lw=1.3,
                ls=":",
                label=r"RS-I $\cos^2\theta\,\mathrm{sinc}^2(\beta)$",
            )
            ax.plot(
                theta_deg,
                p_kirch,
                color=figstyle.PALETTE[4],
                lw=1.1,
                ls=":",
                alpha=0.85,
                label=r"Kirchhoff $\frac{1}{4}(1+\cos\theta)^2\mathrm{sinc}^2(\beta)$",
            )

            panel_letter = chr(ord("a") + idx)
            err_ez_rs1 = float(ff_data["l2_ez_vs_rs1"][idx]) * 100.0
            err_hz_rs2 = float(ff_data["l2_hz_vs_rs2"][idx]) * 100.0
            ax.set_title(
                rf"({panel_letter}) $a / \lambda = {ratio:g}$ "
                rf"($L^2_{{E_z}}={err_ez_rs1:.1f}\%$, $L^2_{{H_z}}={err_hz_rs2:.1f}\%$)"
            )
            ax.set_xlabel(r"Diffraction angle $\theta$ [deg]")
            ax.set_ylabel(
                r"Normalized far-field intensity $U_{\mathrm{2D}}(\theta) / U_{\mathrm{2D}}(0)$ [-]"
            )
            ax.set_xlim(-82.0, 82.0)
            ax.set_ylim(-0.03, 1.08)
            figstyle.spectrum_grid(ax)
            if idx == 0:
                ax.legend(loc="upper right", fontsize=7.4)

        # Panel (c): Wavelength-scale & sub-wavelength slits a/lambda = 1.0 & 0.6 (idx 2, 3)
        ax = axes[2]
        offsets = [0.0, 1.15]
        for local_k, idx in enumerate((2, 3)):
            off = offsets[local_k]
            ratio = float(ratios[idx])
            p_ez = ff_data["u2d_ez_norm"][idx]
            p_hz = ff_data["u2d_hz_norm"][idx]
            p_fraun = ff_data["fraunhofer"][idx]
            p_rs1 = ff_data["rs1"][idx]
            err_ez_rs1 = float(ff_data["l2_ez_vs_rs1"][idx]) * 100.0
            err_hz_rs2 = float(ff_data["l2_hz_vs_rs2"][idx]) * 100.0

            ax.plot(
                theta_deg,
                p_ez + off,
                color=figstyle.PALETTE[0],
                lw=2.0,
                ls="-",
                label=rf"FIT $E_z$, $a/\lambda={ratio:g}$ ($L^2={err_ez_rs1:.1f}\%$)",
            )
            ax.plot(
                theta_deg,
                p_hz + off,
                color=figstyle.PALETTE[1],
                lw=1.9,
                ls="-.",
                label=rf"FIT $H_z$, $a/\lambda={ratio:g}$ ($L^2={err_hz_rs2:.1f}\%$)",
            )
            ax.plot(
                theta_deg,
                p_fraun + off,
                color=figstyle.INK,
                lw=1.2,
                ls="--",
                label="RS-II / Fraunhofer" if local_k == 0 else "_nolegend_",
            )
            ax.plot(
                theta_deg,
                p_rs1 + off,
                color=figstyle.PALETTE[2],
                lw=1.2,
                ls=":",
                label="RS-I obliquity" if local_k == 0 else "_nolegend_",
            )

        ax.set_title(r"(c) Vectorial obliquity split ($a / \lambda = 1.0$ & $0.6$)")
        ax.set_xlabel(r"Diffraction angle $\theta$ [deg]")
        ax.set_ylabel(r"Normalized far-field intensity (offset) [-]")
        ax.set_xlim(-82.0, 82.0)
        ax.set_ylim(-0.04, 2.75)
        figstyle.spectrum_grid(ax)
        ax.legend(loc="upper center", ncol=2, fontsize=6.6)

        figstyle.save(fig, output_path, close=True)


def plot_transmission_and_resolution(
    trans_data: dict,
    res_data: dict,
    output_path: str | Path,
) -> None:
    """Generate the 2-panel figure for transmission tau(a/lambda) and grid convergence."""
    with figstyle.apply_style():
        fig, axes = figstyle.new_figure(nrows=1, ncols=2, width=13.8, height=4.8)

        # Panel (a): Total transmitted power efficiency tau(a/lambda)
        ax = axes[0]
        ratios = trans_data["a_over_lam"]
        ax.plot(
            ratios,
            trans_data["tau_ez_thick"],
            "o-",
            color=figstyle.PALETTE[0],
            lw=2.0,
            ms=5.5,
            label=r"FIT $\mathbf{E} \parallel$ slit ($E_z$), $t = 0.20\lambda$ PEC box",
        )
        ax.plot(
            ratios,
            trans_data["tau_hz_thick"],
            "s-",
            color=figstyle.PALETTE[1],
            lw=2.0,
            ms=5.5,
            label=r"FIT $\mathbf{E} \perp$ slit ($H_z$), $t = 0.20\lambda$ PEC box",
        )
        ax.plot(
            ratios,
            trans_data["tau_ez_sheet"],
            "^--",
            color=figstyle.PALETTE[3],
            lw=1.6,
            ms=5.0,
            label=r"FIT $\mathbf{E} \parallel$ slit ($E_z$), $t = 0$ conformal PEC sheet",
        )
        ax.plot(
            ratios,
            trans_data["tau_hz_sheet"],
            "v--",
            color=figstyle.PALETTE[5],
            lw=1.6,
            ms=5.0,
            label=r"FIT $\mathbf{E} \perp$ slit ($H_z$), $t = 0$ conformal PEC sheet",
        )
        ax.plot(
            trans_data["model_a_over_lam"],
            trans_data["model_tau_ez_thick"],
            color="#004C78",
            ls=":",
            lw=1.4,
            label=r"Waveguide TE$_{10}$ model ($t = 0.20\lambda$)",
        )
        ax.plot(
            trans_data["model_a_over_lam"],
            trans_data["model_tau_hz_thick"],
            color="#8C2D04",
            ls=":",
            lw=1.4,
            label=r"Takakura TEM Fabry–Pérot model ($t = 0.20\lambda$)",
        )
        ax.axhline(
            1.0,
            color="#64748B",
            ls="--",
            lw=1.0,
            label=r"Scalar Kirchhoff limit $\tau = 1$",
        )
        ax.axvline(0.5, color=figstyle.PALETTE[0], ls=":", lw=1.2, alpha=0.8)
        ax.text(
            0.52,
            0.12,
            r"TE$_{10}$ cutoff $a = \lambda/2$",
            color=figstyle.PALETTE[0],
            fontsize=8.5,
        )
        ax.set_xlabel(r"Normalized slit width $a / \lambda$ [-]")
        ax.set_ylabel(
            r"Normalized transmission efficiency $\tau = P'_{\mathrm{trans}} / (a I_{\mathrm{inc}})$ [-]"
        )
        ax.set_title(r"(a) Vectorial transmission $\tau(a/\lambda)$: polarization & thickness")
        ax.set_ylim(-0.02, 1.32)
        figstyle.spectrum_grid(ax)
        ax.legend(loc="lower right", fontsize=7.6)

        # Panel (b): Resolution convergence study
        ax = axes[1]
        cpw = res_data["cpw_values"]
        ax.plot(
            cpw,
            res_data["l2_vs_rs1"] * 100.0,
            "o-",
            color=figstyle.PALETTE[0],
            lw=2.0,
            ms=5.5,
            label=r"Far-field $L^2$ error vs RS-I [%]",
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
        tau_rel = (
            np.abs(res_data["tau_values"] - res_data["tau_values"][-1])
            / res_data["tau_values"][-1]
        )
        ax.plot(
            cpw,
            tau_rel * 100.0,
            "^-.",
            color=figstyle.PALETTE[1],
            lw=1.8,
            ms=5.5,
            label=r"$|\tau(\mathrm{cpw}) - \tau_{\mathrm{fine}}| / \tau_{\mathrm{fine}}$ [%]",
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
        ax.set_title(
            r"(b) Mesh convergence ($a = 2\lambda$, $t = 0.15\lambda$, $\mathbf{E} \parallel$ slit)"
        )
        figstyle.spectrum_grid(ax)
        ax.legend(loc="upper right", fontsize=8.0)

        figstyle.save(fig, output_path, close=True)
