# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

# SPDX-License-Identifier: Apache-2.0

"""
Helper utilities for Tutorial 5.8 (Directional Coupler) addressing kit gaps
to be upstreamed into ``wakis.templates.photonics`` and ``wakis.sources``:

1. ``PhasedGaussianSource``: Gaussian pulse time profile with carrier phase
   offset ``phase`` [rad] and zero DC integral for multi-port phased excitation.
2. ``trapezoidal_effective_width``: Maps 3D slanted-sidewall waveguide
   parameters (``w_gds``, ``dilation``, ``thickness``, ``sidewall_angle_deg``)
   to the 2D EIM mid-plane width and gap without triggering the 2D ``z_min``
   overwrite / unshifted normal-distance limitation in ``PolySlab``.
3. ``solve_coupler_supermodes``: Extracts symmetric (even) and antisymmetric
   (odd) supermode profiles and dispersion ``n_even(lambda), n_odd(lambda)``
   from ``solve_modes``, integrates the S-bend coupling phase contribution
   ``L_sb_eff(lambda)``, and evaluates coupled-mode theory (CMT) spectra.
4. ``run_phased_dual_drive``: Launches multiple ``ModeSource`` ports
   simultaneously with arbitrary carrier phases and extracts normalized output
   power spectra and steady-state 2D DFT fields.
5. ``find_crossing_wavelength``: Sub-grid linear interpolation of the 50:50
   (3 dB) power-splitting wavelength ``lambda_cross`` and total transmittance.
6. ``coupler_mesh_overrides`` / ``build_coupler_mesh_axes``: Builds 2D
   coupling-region ``mesh_overrides`` boxes with per-axis ``(hx, hy, 1.0)`` caps
   (avoiding the 2D ``PICSimulation`` scalar ``max_cell`` gotcha that splits
   ``Nz > 1`` when ``max_cell < h_core``) and measures actual mesh resolution
   metrics (``min_feature_nm``, ``dx_background_nm``, ``dx_min_nm``,
   ``cells_across_feature``, ``n_cells``).
"""

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon as MplPolygon, Rectangle
import numpy as np
from scipy.constants import c as c_light, epsilon_0

from wakis import figstyle, schematic
from wakis.field_monitors import DFTMonitor
from wakis.mode_solver import ModeMonitor, ModeSource, solve_modes
from wakis.templates import meshing
from wakis.templates.photonics import (
    PICSimulation,
    n_to_eps,
    slab_neff,
    vertical_slab_mode,
)
from wakis.templates.simulation import Simulation


class PhasedGaussianSource:
    """
    Zero-DC Gaussian pulse time profile with carrier phase offset ``phase``.

    The time signal is the normalized time derivative of a phase-shifted
    Gaussian-modulated sinusoid:

    .. math::

        s(t) = \\frac{1}{2\\pi f_0}\\frac{\\mathrm{d}}{\\mathrm{d}t}
        \\left[g(\\tau)\\sin(2\\pi f_0 \\tau + \\phi)\\right]
        = g(\\tau)\\left[\\cos(2\\pi f_0 \\tau + \\phi)
        - \\frac{\\tau}{2\\pi f_0 \\sigma_t^2}\\sin(2\\pi f_0 \\tau + \\phi)\\right],

    where :math:`\\tau = t - t_0` and :math:`g(\\tau) = \\exp(-\\tau^2 / 2\\sigma_t^2)`.
    Under the Wakis Fourier convention :math:`\\int s(t) e^{-\\mathrm{i} 2\\pi f t}\\,\\mathrm{d}t`,
    a positive phase :math:`\\phi` multiplies the forward complex mode amplitude
    at :math:`f > 0` by :math:`e^{+\\mathrm{i}\\phi}`.

    Parameters
    ----------
    base_pulse : GaussianSource
        Reference pulse providing ``frequency``, ``fwidth``, ``sigma_t``,
        ``t0``, ``start_time``, and ``end_time``.
    phase : float, optional
        Carrier phase offset :math:`\\phi` [rad]. Default is ``0.0``.
    """

    def __init__(self, base_pulse, phase=0.0):
        self.frequency = float(base_pulse.frequency)
        self.fwidth = float(base_pulse.fwidth)
        self.sigma_t = float(base_pulse.sigma_t)
        self.t0 = float(base_pulse.t0)
        self.start_time = float(base_pulse.start_time)
        self.end_time = float(base_pulse.end_time)
        self.phase = float(phase)

    def __call__(self, t):
        tau = t - self.t0
        w = 2.0 * np.pi * self.frequency
        arg = w * tau + self.phase
        envelope = np.exp(-(tau**2) / (2.0 * self.sigma_t**2))
        signal = envelope * (
            np.cos(arg) - tau / (w * self.sigma_t**2) * np.sin(arg)
        )
        return np.where((t >= self.start_time) & (t <= self.end_time), signal, 0.0)


def trapezoidal_effective_width(
    w_gds,
    dilation=0.0,
    thickness=220e-9,
    sidewall_angle_deg=0.0,
    pitch=0.55e-6,
):
    """
    Compute bottom, top, and 2D mid-height effective waveguide widths and gaps
    for a trapezoidal SOI strip waveguide with bottom reference plane.

    Parameters
    ----------
    w_gds : float
        Nominal mask layout path width [m].
    dilation : float, optional
        Mask bias / dilation added to bottom width [m]. Default is ``0.0``.
    thickness : float, optional
        Waveguide core thickness :math:`h_{\\mathrm{wg}}` [m]. Default is ``220e-9``.
    sidewall_angle_deg : float, optional
        Sidewall tilt angle :math:`\\theta_{\\mathrm{sw}}` from vertical [deg].
        Default is ``0.0``.
    pitch : float, optional
        Center-to-center waveguide pitch in the coupling region [m].
        Default is ``0.55e-6``.

    Returns
    -------
    dict of float
        Dictionary with keys ``w_bot``, ``w_top``, ``w_eff``, ``gap_bot``,
        ``gap_top``, and ``gap_eff`` [m].
    """
    w_bot = float(w_gds) + float(dilation)
    tan_sw = float(np.tan(np.deg2rad(float(sidewall_angle_deg))))
    h = float(thickness)
    w_top = w_bot - 2.0 * h * tan_sw
    w_eff = 0.5 * (w_bot + w_top)
    p = float(pitch)
    return {
        "w_bot": w_bot,
        "w_top": w_top,
        "w_eff": w_eff,
        "gap_bot": p - w_bot,
        "gap_top": p - w_top,
        "gap_eff": p - w_eff,
    }


def coupler_mesh_overrides(
    L_c,
    gap_eff,
    w_eff,
    n_cells_gap,
    dx_sb=6.0e-6,
    dy_sb=1.725e-6,
):
    """
    Build a ``mesh_overrides`` box covering the parallel coupling section and
    the S-bend transitions where the local gap is below :math:`3 g_{\\mathrm{eff}}`.

    For a cosine S-bend with lateral offset :math:`\\Delta y_{\\mathrm{sb}}` over
    axial length :math:`\\Delta x_{\\mathrm{sb}}`, the local inter-waveguide gap is

    .. math::

        g(u) = g_{\\mathrm{eff}} + \\Delta y_{\\mathrm{sb}}\\,\\left[1 - \\cos(\\pi u)\\right],
        \\qquad u = \\frac{|x| - L_c/2}{\\Delta x_{\\mathrm{sb}}} \\in [0, 1].

    Setting :math:`g(u_{3\\times}) = 3 g_{\\mathrm{eff}}` yields
    :math:`u_{3\\times} = \\pi^{-1}\\arccos(\\max(-1, 1 - 2 g_{\\mathrm{eff}} / \\Delta y_{\\mathrm{sb}}))`
    and :math:`x_{\\mathrm{box}} = L_c / 2 + u_{3\\times}\\Delta x_{\\mathrm{sb}}`.
    Transversely, :math:`y_{\\mathrm{box}} = 1.5 g_{\\mathrm{eff}} + w_{\\mathrm{eff}}`
    encloses the gap and both coupled waveguide cores without placing a grading
    seam inside the core.

    .. note::
       In 2D mode (``two_d=True``), ``PICSimulation`` sets the periodic slab
       domain along :math:`z` to thickness ``h_core``. Passing a scalar
       ``max_cell = h_max`` broadcasts to ``(h_max, h_max, h_max)`` and splits
       ``Nz > 1`` whenever ``h_max < h_core``. Always pass a 3-tuple
       ``(h_max, h_max, 1.0)`` so ``Nz = 1`` is preserved.

    Parameters
    ----------
    L_c : float
        Straight parallel coupling length [m].
    gap_eff : float
        2D mid-height effective coupling gap :math:`g_{\\mathrm{eff}}` [m].
    w_eff : float
        2D mid-height effective waveguide width :math:`w_{\\mathrm{eff}}` [m].
    n_cells_gap : int or float or None
        Target minimum number of cells :math:`N` across ``gap_eff``. If ``None``
        or ``<= 0``, returns ``[]`` (no mesh overrides).
    dx_sb, dy_sb : float, optional
        S-bend axial span and lateral offset [m].

    Returns
    -------
    list of tuple
        ``[((lo_x, lo_y, -inf), (hi_x, hi_y, +inf), (h_max, h_max, 1.0))]`` or ``[]``.
    """
    if n_cells_gap is None or float(n_cells_gap) <= 0.0:
        return []
    g = float(gap_eff)
    w = float(w_eff)
    h_max = g / float(n_cells_gap)
    cos_arg = max(-1.0, min(1.0, 1.0 - 2.0 * g / float(dy_sb)))
    u_3x = float(np.arccos(cos_arg) / np.pi)
    x_box = 0.5 * float(L_c) + u_3x * float(dx_sb)
    y_box = 1.5 * g + w
    return [
        (
            (-x_box, -y_box, -np.inf),
            (+x_box, +y_box, +np.inf),
            (h_max, h_max, 1.0),
        )
    ]


def extract_mesh_metrics_from_grid(x, y, gap_eff, Nz=1):
    """
    Compute actual mesh resolution metrics from 1D node arrays ``x`` and ``y``.

    Parameters
    ----------
    x, y : ndarray
        1D grid node coordinates along :math:`x` and :math:`y` [m].
    gap_eff : float
        Smallest feature size (2D effective coupling gap :math:`g_{\\mathrm{eff}}`) [m].
    Nz : int, optional
        Number of cells along :math:`z` (``1`` in 2D mode).

    Returns
    -------
    dict
        Dictionary containing ``min_feature_nm``, ``dx_background_nm``,
        ``dx_min_nm``, ``cells_across_feature``, ``n_cells``, and ``grid_shape``.
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    dx = np.diff(x_arr)
    dy = np.diff(y_arr)
    g = float(gap_eff)
    mask_gap = (y_arr[:-1] < 0.5 * g - 1e-12) & (y_arr[1:] > -0.5 * g + 1e-12)
    if not np.any(mask_gap):
        ymid = 0.5 * (y_arr[:-1] + y_arr[1:])
        mask_gap = np.arange(len(dy)) == int(np.argmin(np.abs(ymid)))
    h_feat_max = float(np.max(dy[mask_gap]))
    nx, ny, nz = int(len(dx)), int(len(dy)), int(Nz)
    return {
        "min_feature_nm": float(g * 1e9),
        "dx_background_nm": float(max(np.max(dx), np.max(dy)) * 1e9),
        "dx_min_nm": float(min(np.min(dx), np.min(dy)) * 1e9),
        "cells_across_feature": float(g / h_feat_max),
        "n_cells": int(nx * ny * nz),
        "grid_shape": [nx, ny, nz],
    }


def build_coupler_mesh_axes(
    structures,
    domain,
    gap_eff,
    cpw,
    mesh_overrides=(),
    wavelength_min=1.40e-6,
    wavelength_max=1.70e-6,
    wavelength_ref=1.53548e-6,
    n_core=3.48,
    n_clad=1.00,
    n_sub=1.45,
    thickness=0.22e-6,
    n_pml=8,
):
    """
    Build the 1D tensor-product mesh axes ``(x, y)`` of the 2D EIM directional
    coupler without allocating ``GridFIT3D`` / ``SolverFIT3D`` or running
    subpixel ``PolySlab.fill`` (fast preflight/test helper).

    Parameters
    ----------
    structures : list of PolySlab
        2D coupler structures.
    domain : tuple of 4 float
        ``(xmin, xmax, ymin, ymax)`` [m].
    gap_eff : float
        2D effective coupling gap :math:`g_{\\mathrm{eff}}` [m].
    cpw : float
        Background ``cells_per_wavelength``.
    mesh_overrides : sequence of tuple, optional
        Mesh-override boxes passed to ``Simulation``.

    Returns
    -------
    x, y : ndarray
        1D grid node coordinates [m].
    metrics : dict
        Resolution metrics from ``extract_mesh_metrics_from_grid``.
    """
    neff_slab = slab_neff(
        n_core, n_clad, thickness, wavelength_ref, n_sub=n_sub, polarization="TE"
    )
    eps_slab = n_to_eps(neff_slab)
    for s in structures:
        s.material = eps_slab
    h_core = float(wavelength_min) / (neff_slab * float(cpw))
    dom3d = (
        float(domain[0]),
        float(domain[1]),
        float(domain[2]),
        float(domain[3]),
        -0.5 * h_core,
        +0.5 * h_core,
    )
    sim = Simulation(
        c_light / float(wavelength_max),
        c_light / float(wavelength_min),
        structures=structures,
        boundaries=(["cpml", "cpml", "periodic"], ["cpml", "cpml", "periodic"]),
        domain=dom3d,
        cells_per_wavelength=float(cpw),
        n_pml=int(n_pml),
        mesh_overrides=list(mesh_overrides),
    )
    x = sim._axis(0)
    y = sim._axis(1)
    metrics = extract_mesh_metrics_from_grid(x, y, gap_eff=gap_eff, Nz=1)
    return x, y, metrics


def solve_coupler_supermodes(
    pic,
    wavelengths,
    x_coupled=0.0,
    x_sb_start=2.0e-6,
    dx_sb=6.0e-6,
    dy_sb=1.725e-6,
    L_c=4.0e-6,
    window_y=3.8e-6,
    wavelength_profile=1.53e-6,
):
    """
    Compute the symmetric (even) and antisymmetric (odd) supermodes of the
    coupled waveguide section and integrate the S-bend coupling contribution
    directly on ``pic.solver``.

    Parameters
    ----------
    pic : PICSimulation
        Built 2D EIM ``PICSimulation`` of the directional coupler.
    wavelengths : array_like
        Wavelengths :math:`\\lambda` [m] at which to evaluate supermode dispersion.
    x_coupled : float, optional
        Axial coordinate :math:`x` [m] in the parallel coupling region.
    x_sb_start : float, optional
        Start coordinate :math:`x` [m] of the output S-bend.
    dx_sb, dy_sb : float, optional
        Axial length and lateral offset [m] of each S-bend arm.
    L_c : float, optional
        Straight parallel coupling length [m].
    window_y : float, optional
        Transverse port-plane span [m] enclosing both waveguides.
    wavelength_profile : float, optional
        Reference wavelength [m] at which 1D transverse supermode profiles
        :math:`E_y^{\\mathrm{even}}(y)` and :math:`E_y^{\\mathrm{odd}}(y)` are returned.

    Returns
    -------
    dict of ndarray
        Supermode profiles, effective indices ``n_even``, ``n_odd``, index
        splitting ``dn_eff``, crossover beat length ``L_pi``, S-bend effective
        coupling length ``L_sb_eff``, CMT cross/bar transmissions, and CMT
        50:50 crossing wavelength ``wl_cross_cmt``.
    """
    wls = np.asarray(wavelengths, dtype=float)
    z_mid = 0.5 * (pic.grid.zmin + pic.grid.zmax)

    # 1. Transverse supermode profiles at wavelength_profile
    f_prof = c_light / float(wavelength_profile)
    modes_prof = solve_modes(
        pic.solver,
        f_prof,
        center=(float(x_coupled), 0.0, z_mid),
        size=(0.0, float(window_y), np.inf),
        n_modes=2,
    )
    y_coords = modes_prof[0].coords["Eb"][0]
    mask_y = np.abs(y_coords) <= 1.2e-6
    ey_even = np.real(modes_prof[0].e["b"][mask_y, 0]).copy()
    ey_odd = np.real(modes_prof[1].e["b"][mask_y, 0]).copy()
    y_prof = y_coords[mask_y].copy()

    # Orient even mode positive on both arms, odd mode positive on upper arm (+y)
    if np.mean(ey_even) < 0:
        ey_even *= -1.0
    if np.mean(ey_odd[y_prof > 0]) < 0:
        ey_odd *= -1.0

    # 2. Wavelength dispersion at x_coupled and 5 midpoint quadrature stations on S-bend
    u_pts = np.array([0.05, 0.15, 0.25, 0.35, 0.45])
    du = 0.10
    x_sb_pts = float(x_sb_start) + float(dx_sb) * u_pts
    dydx_pts = (0.5 * np.pi * float(dy_sb) / float(dx_sb)) * np.sin(np.pi * u_pts)
    ds_dx_pts = np.hypot(1.0, dydx_pts)

    n_even = np.empty_like(wls)
    n_odd = np.empty_like(wls)
    dn_sb_int = np.zeros_like(wls)

    for i, wl in enumerate(wls):
        f = c_light / float(wl)
        m0 = solve_modes(
            pic.solver,
            f,
            center=(float(x_coupled), 0.0, z_mid),
            size=(0.0, float(window_y), np.inf),
            n_modes=2,
        )
        ne = float(np.real(m0[0].neff))
        no = float(np.real(m0[1].neff))
        n_even[i] = max(ne, no)
        n_odd[i] = min(ne, no)

        acc = 0.0
        for xs, dsdx in zip(x_sb_pts, ds_dx_pts):
            ms = solve_modes(
                pic.solver,
                f,
                center=(float(xs), 0.0, z_mid),
                size=(0.0, float(window_y) + 0.8e-6, np.inf),
                n_modes=2,
            )
            dn_s = abs(float(np.real(ms[0].neff - ms[1].neff)))
            acc += 2.0 * float(dx_sb) * du * float(dsdx) * dn_s
        dn_sb_int[i] = acc

    dn_eff = np.maximum(n_even - n_odd, 1e-12)
    L_pi = wls / (2.0 * dn_eff)
    L_sb_eff = dn_sb_int / dn_eff
    theta_cmt = 0.5 * np.pi * (float(L_c) + L_sb_eff) / L_pi
    t_cross_cmt = np.sin(theta_cmt) ** 2
    t_bar_cmt = np.cos(theta_cmt) ** 2
    wl_cross_cmt, _ = find_crossing_wavelength(wls, t_cross_cmt, t_bar_cmt)

    return {
        "supermode_y": y_prof.astype(np.float32),
        "supermode_ey_even": ey_even.astype(np.float32),
        "supermode_ey_odd": ey_odd.astype(np.float32),
        "supermode_neff_even_ref": np.float64(np.real(modes_prof[0].neff)),
        "supermode_neff_odd_ref": np.float64(np.real(modes_prof[1].neff)),
        "cmt_wavelengths": wls.astype(np.float64),
        "cmt_n_even": n_even.astype(np.float64),
        "cmt_n_odd": n_odd.astype(np.float64),
        "cmt_dn_eff": dn_eff.astype(np.float64),
        "cmt_L_pi": L_pi.astype(np.float64),
        "cmt_L_sb_eff": L_sb_eff.astype(np.float64),
        "cmt_t_cross": t_cross_cmt.astype(np.float64),
        "cmt_t_bar": t_bar_cmt.astype(np.float64),
        "cmt_wl_cross": np.float64(wl_cross_cmt),
    }


def run_phased_dual_drive(
    pic,
    port_phases,
    energy_decay=1e-4,
    max_time=None,
    collect_dft=True,
):
    """
    Run a time-domain FIT simulation with simultaneous phase-offset ``ModeSource``
    excitations on multiple input ports.

    Parameters
    ----------
    pic : PICSimulation
        Configured ``PICSimulation`` instance.
    port_phases : dict of {str: float}
        Mapping from driven port name to carrier phase offset [rad], e.g.
        ``{"in_bot": 0.0, "in_top": -0.5 * np.pi}``.
    energy_decay : float, optional
        Energy ring-down termination threshold. Default is ``1e-4``.
    max_time : float, optional
        Maximum simulation duration [s].
    collect_dft : bool, optional
        If True, record 2D midplane DFT fields at ``pic.field_wavelengths``.

    Returns
    -------
    dict
        Normalized output port power spectra ``power[port_name]``, complex
        amplitudes, and optional DFT field arrays.
    """
    if max_time is None:
        lx = pic.grid.xmax - pic.grid.xmin
        ly = pic.grid.ymax - pic.grid.ymin
        transit = np.hypot(lx, ly) * pic.n_core / c_light
        max_time = pic.sim.pulse.end_time + 15.0 * transit

    if pic.solver is None:
        pic.build()
    else:
        pic.solver.reset_fields()
    sources = []
    inward_map = {}
    for pname, phase in port_phases.items():
        port = pic.port_map[pname]
        in_center, in_size, in_inward = pic._port_plane_args(port)
        inward_map[pname] = in_inward
        a_in = port.axis
        nodes_a = pic.grid.x if a_in == 0 else pic.grid.y
        k_mon = int(np.argmin(np.abs(nodes_a[:-1] - in_center[a_in])))
        k_src = int(np.clip(k_mon - 2 * in_inward, 2, len(nodes_a) - 3))
        src_center = in_center.copy()
        src_center[a_in] = nodes_a[k_src]
        pulse_phased = PhasedGaussianSource(pic.sim.pulse, phase=phase)
        sources.append(
            ModeSource(
                pulse_phased,
                center=src_center,
                size=in_size,
                direction="+" if in_inward > 0 else "-",
                mode_index=port.mode_index,
                mode_frequency=pic.f_center,
            )
        )

    port_monitors, monitors_list = {}, []
    for p in pic.ports:
        p_center, p_size, p_inw = pic._port_plane_args(p)
        inward_map[p.name] = p_inw
        mon = ModeMonitor(
            pic.frequencies,
            center=p_center,
            size=p_size,
            n_modes=p.mode_index + 1,
            mode_frequency=pic.f_center,
        )
        port_monitors[p.name] = mon
        monitors_list.append(mon)

    dft_mon = None
    stride = max(1, int(np.ceil(max(pic.grid.Nx, pic.grid.Ny) / 400)))
    if collect_dft and pic.field_wavelengths:
        f_dft = [c_light / w for w in pic.field_wavelengths]
        z_mid = 0.5 * (pic.grid.zmin + pic.grid.zmax)
        dft_mon = DFTMonitor(
            f_dft,
            components=("Ex", "Ey"),
            center=(
                0.5 * (pic.grid.xmin + pic.grid.xmax),
                0.5 * (pic.grid.ymin + pic.grid.ymax),
                z_mid,
            ),
            size=(np.inf, np.inf, 0.0),
        )
        monitors_list.append(dft_mon)

    Nt = int(np.ceil(max_time / pic.solver.dt))
    pic.solver.emsolve(
        Nt,
        source=sources,
        monitors=monitors_list,
        stop_when_energy_decayed=energy_decay,
    )

    p_inc_total = np.zeros_like(pic.frequencies, dtype=float)
    inc_amps = {}
    for pname in port_phases:
        port = pic.port_map[pname]
        a_p, a_m = port_monitors[pname].get_amplitudes(port.mode_index)
        a_inc = a_p if inward_map[pname] > 0 else a_m
        inc_amps[pname] = a_inc
        p_inc_total += np.abs(a_inc) ** 2

    out_power = {}
    out_amp = {}
    for p in pic.ports:
        a_p, a_m = port_monitors[p.name].get_amplitudes(p.mode_index)
        b_out = a_m if inward_map[p.name] > 0 else a_p
        out_amp[p.name] = b_out
        out_power[p.name] = np.abs(b_out) ** 2 / np.maximum(p_inc_total, 1e-30)

    dft_out = {}
    if dft_mon is not None:
        nx, ny = pic.grid.Nx, pic.grid.Ny
        xc = 0.5 * (pic.grid.x[:-1] + pic.grid.x[1:])
        yc = 0.5 * (pic.grid.y[:-1] + pic.grid.y[1:])
        ex_raw, _ = dft_mon.get_field("Ex")
        ey_raw, _ = dft_mon.get_field("Ey")
        ex_c = ex_raw[:, :nx, :ny, 0]
        ey_c = ey_raw[:, :nx, :ny, 0]
        x_dft = xc[::stride].astype(np.float32)
        y_dft = yc[::stride].astype(np.float32)
        for idx_w, w_val in enumerate(pic.field_wavelengths):
            ex = ex_c[idx_w, ::stride, ::stride].astype(np.complex64)
            ey = ey_c[idx_w, ::stride, ::stride].astype(np.complex64)
            dft_out[w_val] = {
                "x": x_dft,
                "y": y_dft,
                "Ex": ex,
                "Ey": ey,
                "E2": (np.abs(ex) ** 2 + np.abs(ey) ** 2).astype(np.float32),
            }

    return {
        "power": out_power,
        "out_amp": out_amp,
        "inc_amp": inc_amps,
        "dft": dft_out,
    }


def find_crossing_wavelength(wavelengths, t_cross, t_bar):
    """
    Interpolate the 50:50 power-splitting wavelength ``lambda_cross`` where
    ``t_cross(lambda) == t_bar(lambda)`` and the total transmittance there.

    Parameters
    ----------
    wavelengths : array_like
        Wavelength array [m].
    t_cross, t_bar : array_like
        Cross-port and bar-port power transmission arrays.

    Returns
    -------
    wl_cross : float
        Interpolated 50:50 crossing wavelength [m].
    t_tot_cross : float
        Total guided power ``t_cross + t_bar`` at ``wl_cross``.
    """
    w = np.asarray(wavelengths, dtype=float)
    tc = np.asarray(t_cross, dtype=float)
    tb = np.asarray(t_bar, dtype=float)
    order = np.argsort(w)
    w, tc, tb = w[order], tc[order], tb[order]
    diff = tc - tb
    tot = tc + tb

    crossings = np.flatnonzero(diff[:-1] * diff[1:] <= 0.0)
    if crossings.size > 0:
        # Pick crossing closest to band center
        w_mid = 0.5 * (w[0] + w[-1])
        best_k = int(crossings[np.argmin(np.abs(w[crossings] - w_mid))])
        d0, d1 = diff[best_k], diff[best_k + 1]
        alpha = 0.0 if np.isclose(d1, d0) else float(-d0 / (d1 - d0))
        alpha = float(np.clip(alpha, 0.0, 1.0))
        wl_c = float(w[best_k] + alpha * (w[best_k + 1] - w[best_k]))
        t_tot_c = float(tot[best_k] + alpha * (tot[best_k + 1] - tot[best_k]))
        return wl_c, t_tot_c

    k_min = int(np.argmin(np.abs(diff)))
    return float(w[k_min]), float(tot[k_min])


def plot_asymmetric_eim_stack(
    n_core=3.48,
    n_clad=1.0,
    n_sub=1.45,
    thickness=220e-9,
    wavelength_ref=1.53e-6,
    wavelengths=None,
    out_path=None,
):
    """
    Plot an asymmetric 3-layer vertical SOI stack (``n_sub != n_clad``), its 1D
    :math:`\\mathrm{TE}_0` slab mode profile, and :math:`n_{\\mathrm{eff,slab}}(\\lambda)`.
    """
    if wavelengths is None:
        wavelengths = np.linspace(1400e-9, 1700e-9, 51)
    vm = vertical_slab_mode(
        n_core=n_core,
        n_clad=n_clad,
        n_sub=n_sub,
        thickness=thickness,
        wavelength=wavelength_ref,
    )
    neff_band = np.array(
        [
            slab_neff(n_core, n_clad, thickness, w, n_sub=n_sub, polarization="TE")
            for w in wavelengths
        ]
    )

    figstyle.apply_style()
    fig, axes = figstyle.new_figure(nrows=1, ncols=3, width=11.6, height=3.8)
    d_nm = thickness * 1e9
    ax0, ax1, ax2 = axes
    ax0.axhspan(-600, -0.5 * d_nm, color="#d9e6f2", label=f"SiO₂ BOX (n={n_sub:.2f})")
    ax0.axhspan(
        -0.5 * d_nm,
        0.5 * d_nm,
        color="#2b5c8f",
        alpha=0.85,
        label=f"Si core ({d_nm:.0f} nm, n={n_core:.2f})",
    )
    ax0.axhspan(0.5 * d_nm, 600, color="#f4f8fc", label=f"Air clad (n={n_clad:.2f})")
    ax0.axhline(-0.5 * d_nm, color="#16324f", lw=1.2)
    ax0.axhline(0.5 * d_nm, color="#16324f", lw=1.2)
    ax0.set(
        ylim=(-600, 600),
        xlim=(0, 1),
        xticks=[],
        ylabel="Vertical coordinate z − h/2 [nm]",
        title="(a) Asymmetric SOI layer stack",
    )
    ax0.legend(loc="upper right", fontsize=8)

    z_nm = vm["z"] * 1e9
    ax1.axvspan(-0.5 * d_nm, 0.5 * d_nm, color="#2b5c8f", alpha=0.18, label="Si core")
    ax1.plot(z_nm, vm["field"], color="#d95f02", lw=2.0, label=r"$|E_y(z)|$ (TE$_0$)")
    ax1.plot(
        z_nm,
        vm["field"] ** 2,
        color="#1b9e77",
        lw=1.6,
        ls="--",
        label=f"$|E_y|^2$ (Γ={vm['confinement'] * 100:.1f}%)",
    )
    ax1.set(
        xlim=(-600, 600),
        ylim=(0, 1.08),
        xlabel="Vertical coordinate z − h/2 [nm]",
        ylabel="Normalized mode amplitude",
        title=f"(b) 1D slab TE₀ mode (n_eff = {vm['neff']:.4f} @ {wavelength_ref * 1e9:.0f} nm)",
    )
    ax1.grid(True, ls=":", alpha=0.4)
    ax1.legend(loc="upper right", fontsize=8)

    wl_nm = np.asarray(wavelengths) * 1e9
    ax2.plot(
        wl_nm,
        neff_band,
        color="#2b5c8f",
        lw=2.0,
        label=r"$n_{\mathrm{eff,slab}}(\lambda)$",
    )
    ax2.axhline(
        vm["neff"],
        color="#d95f02",
        ls="--",
        lw=1.6,
        label=f"Constant @ {wavelength_ref * 1e9:.0f} nm ({vm['neff']:.4f})",
    )
    ax2.set(
        xlabel="Wavelength λ [nm]",
        ylabel=r"Vertical slab effective index $n_{\mathrm{eff,slab}}$",
        title="(c) Slab effective index dispersion",
    )
    ax2.grid(True, ls=":", alpha=0.4)
    ax2.legend(loc="upper right", fontsize=8)

    if out_path is not None:
        figstyle.save(fig, out_path, close=True, auto_legend_below=False)
    return fig, axes


def build_coupler_graded_axes(
    domain=(-8.65e-6, 8.65e-6, -3.00e-6, 3.00e-6),
    L_c: float = 4.00e-6,
    gap_eff: float = 207.02e-9,
    w_eff: float = 342.98e-9,
    y_port: float = 2.00e-6,
    dx_sb: float = 6.00e-6,
    dy_sb: float = 1.725e-6,
    cpw: float = 20.0,
    n_cells_gap: int = 8,
    n_pml: int = 10,
):
    """
    Construct mirror-symmetric graded non-uniform Cartesian node arrays
    ``(x, y, z)`` (``x == -x[::-1]``, ``y == -y[::-1]``) for the 2D EIM
    directional coupler using ``wakis.templates.meshing.graded_axis``:

    - Transversely (``y``): snaps exact grid nodes to the coupling-section gap
      walls ``y = +-gap_eff/2``, outer coupled core walls ``y = +-(gap_eff/2 + w_eff)``,
      and input/output port core walls ``y = +-(y_port +- w_eff/2)``; places
      fine cells across the coupling slot (``dy_gap <= gap_eff / max(n_cells_gap, 10)``
      at ``cpw=20``) and waveguide cores (``dy_core ~ 24.5 nm``), while smoothly
      grading across the S-bend transition corridor (``dy <= 34 nm``) and outer
      air cladding (``|y| > y_port + w_eff/2``, ``dy <= 52 nm``).
    - Axially (``x``): uses non-square cells elongated along the propagation
      axis (``dx > dy_min``), with ``dx ~ 28 nm`` in the strong-coupling window
      ``|x| <= L_c/2 + u_3x * dx_sb``, ``dx ~ 38 nm`` across the uncoupled outer
      S-bend transition, and ``dx ~ 26 nm`` across the port planes and CPML leads.

    Returns
    -------
    x, y, z : ndarray
        1D grid node coordinates [m] including ``n_pml`` uniform CPML layers.
    metrics : dict
        Resolution metrics from ``extract_mesh_metrics_from_grid``.
    """
    scale = 20.0 / max(float(cpw), 8.0)
    g = float(gap_eff)
    w = float(w_eff)
    yp = float(y_port)
    x_max = max(abs(float(domain[0])), abs(float(domain[1])))
    y_max = max(abs(float(domain[2])), abs(float(domain[3])))

    n_half_gap = max(int(np.ceil(0.5 * float(n_cells_gap or 4))), 2)
    if float(cpw) >= 18.0 and (n_cells_gap is None or int(n_cells_gap) == 8):
        n_half_gap = 5  # 10 exact symmetric cells across gap_eff at L3
    dy_gap = g / (2.0 * n_half_gap)
    dy_core = min(w / 14.0, 24.5e-9 * scale)
    dy_sb_cell = 34.0e-9 * scale
    dy_clad = 52.0e-9 * scale

    y_pts = [0.0, 0.5 * g, 0.5 * g + w, yp - 0.5 * w, yp + 0.5 * w, y_max]
    y_sizes = [dy_gap, dy_core, dy_sb_cell, dy_core, dy_clad]
    y_min_c = [n_half_gap, max(int(round(w / dy_core)), 6), 4, max(int(round(w / dy_core)), 6), 4]
    y_pos = meshing.graded_axis(y_pts, y_sizes, y_min_c, grading=1.08)
    y_dom = np.concatenate([-y_pos[:0:-1], y_pos])
    dy_pml = float(y_dom[-1] - y_dom[-2])
    y_full = np.concatenate(
        [
            y_dom[0] - dy_pml * np.arange(int(n_pml), 0, -1),
            y_dom,
            y_dom[-1] + dy_pml * np.arange(1, int(n_pml) + 1),
        ]
    )

    cos_arg = max(-1.0, min(1.0, 1.0 - 2.0 * g / float(dy_sb)))
    u_3x = float(np.arccos(cos_arg) / np.pi)
    x_box = min(0.5 * float(L_c) + u_3x * float(dx_sb), x_max - 1.5e-6)
    x_port_zone = max(x_box + 0.5e-6, x_max - 0.90e-6)

    dx_coupled = 28.0e-9 * scale
    dx_sb_max = 38.0e-9 * scale
    dx_port = 26.0e-9 * scale
    x_pts = [0.0, x_box, x_port_zone, x_max]
    x_sizes = [dx_coupled, dx_sb_max, dx_port]
    x_min_c = [10, 10, 6]
    x_pos = meshing.graded_axis(x_pts, x_sizes, x_min_c, grading=1.06)
    x_dom = np.concatenate([-x_pos[:0:-1], x_pos])
    dx_pml = float(x_dom[-1] - x_dom[-2])
    x_full = np.concatenate(
        [
            x_dom[0] - dx_pml * np.arange(int(n_pml), 0, -1),
            x_dom,
            x_dom[-1] + dx_pml * np.arange(1, int(n_pml) + 1),
        ]
    )

    if len(domain) >= 6:
        z_full = np.array([float(domain[4]), float(domain[5])], dtype=float)
    else:
        h_z = max(dy_gap, 24.5e-9 * scale)
        z_full = np.array([-0.5 * h_z, +0.5 * h_z], dtype=float)
    metrics = extract_mesh_metrics_from_grid(x_full, y_full, gap_eff=g, Nz=1)
    return x_full, y_full, z_full, metrics


class NonUniformCouplerPIC(PICSimulation):
    """
    ``PICSimulation`` subclass that discretizes the 2D EIM directional coupler
    on the mirror-symmetric graded non-uniform Cartesian mesh from
    ``build_coupler_graded_axes`` (fine transverse ``dy`` across the coupling
    gap and waveguide cores, coarser ``dy`` in the S-bend trench and outer air
    cladding, and non-square ``dx > dy_min`` along the propagation axis).
    """

    def __init__(
        self,
        wavelength_min: float,
        wavelength_max: float,
        L_c: float = 4.00e-6,
        gap_eff: float = 207.02e-9,
        w_eff: float = 342.98e-9,
        n_cells_gap: int = 8,
        y_port: float = 2.00e-6,
        dx_sb: float = 6.00e-6,
        dy_sb: float = 1.725e-6,
        **kwargs,
    ):
        self._nu_L_c = float(L_c)
        self._nu_gap_eff = float(gap_eff)
        self._nu_w_eff = float(w_eff)
        self._nu_n_cells_gap = int(n_cells_gap if n_cells_gap is not None else 8)
        self._nu_y_port = float(y_port)
        self._nu_dx_sb = float(dx_sb)
        self._nu_dy_sb = float(dy_sb)
        kwargs["mesh_overrides"] = ()
        super().__init__(wavelength_min, wavelength_max, **kwargs)

    def build(self):
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
        x, y, z, _ = build_coupler_graded_axes(
            domain=domain,
            L_c=self._nu_L_c,
            gap_eff=self._nu_gap_eff,
            w_eff=self._nu_w_eff,
            y_port=self._nu_y_port,
            dx_sb=self._nu_dx_sb,
            dy_sb=self._nu_dy_sb,
            cpw=self.cells_per_wavelength,
            n_cells_gap=self._nu_n_cells_gap,
            n_pml=self.sim.n_pml,
        )
        axes_xyz = (x, y, z)
        self.sim._axis = lambda axis: axes_xyz[int(axis)]
        self.solver = self.sim.build()
        self.solver.ieps[:, :, :, "z"] = 0.0
        self.grid = self.sim.grid
        self.frequencies = self.sim.frequencies
        self.wavelengths = c_light / self.frequencies
        return self.solver


class _PolyStructProxy:
    """Lightweight polygon wrapper without ``bounds`` so ``wakis.schematic`` treats 2D ``PolySlab`` arms as polygons."""

    def __init__(self, vertices, material, name: str):
        self.vertices = np.asarray(vertices, dtype=float)
        self.material = material
        self.name = name


class _SchematicSolverProxy:
    """Proxy around ``SolverFIT3D`` hiding ``ieps`` so ``wakis.schematic`` uses explicit polygon structures."""

    def __init__(self, solver, eps_clad: float):
        self._solver = solver
        self.grid = solver.grid
        self.Nx = solver.Nx
        self.Ny = solver.Ny
        self.Nz = solver.Nz
        self.dt = solver.dt
        self.n_pml = getattr(solver, "n_pml", 10)
        self.bc_low = getattr(solver, "bc_low", ["cpml", "cpml", "periodic"])
        self.bc_high = getattr(solver, "bc_high", ["cpml", "cpml", "periodic"])
        self.bg = [float(eps_clad), 1.0, 0.0]
        self._assigned_structures = ()


def _build_coupler_schematic_io(pic, quadrature: bool = False, include_flux_box: bool = True):
    """Build normalized ``(sources, monitors)`` dicts for ``wakis.schematic``."""
    xmin, xmax = float(pic.grid.xmin), float(pic.grid.xmax)
    ymin, ymax = float(pic.grid.ymin), float(pic.grid.ymax)
    dx_bg = float(np.median(np.diff(pic.grid.x)))
    dy_bg = float(np.median(np.diff(pic.grid.y)))
    pulse_obj = pic.sim.pulse
    f0 = float(pulse_obj.frequency)
    fwidth = float(pulse_obj.fwidth)
    freqs = np.asarray(pic.frequencies, dtype=float)
    f_lo_thz = float(np.min(freqs)) * 1e-12
    f_hi_thz = float(np.max(freqs)) * 1e-12
    freq_str = f"{freqs.size} freqs in [{f_lo_thz:.1f}, {f_hi_thz:.1f}] THz"

    sources_s = []
    monitors_s = []
    for pt in pic.ports:
        cx, cy, _ = (float(v) for v in pt.center)
        _, sy, _ = (float(v) for v in pt.size)
        hy = min(0.5 * sy, 0.72e-6)
        v_b = (cy - hy, cy + hy)
        pos_s = f"x = {cx * 1e6:.2f} µm, y = {cy * 1e6:.2f} µm"
        if pt.name == "in_bot":
            x_src = cx - 0.12e-6
            lbl_src = (
                "in_bot ModeSource (TE₀, φ = 0)"
                if quadrature
                else "in_bot ModeSource (TE₀, +x)"
            )
            sources_s.append(
                {
                    "kind": "sheet",
                    "type_name": "ModeSource",
                    "short_word": "TE₀ mode",
                    "desc": f"{lbl_src} ({pos_s})",
                    "label": lbl_src,
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
            monitors_s.append(
                {
                    "kind": "plane",
                    "type_name": "ModeMonitor",
                    "short_word": "in_bot",
                    "desc": f"in_bot ModeMonitor (S₁₁, {pos_s})",
                    "label": "Port 1 'in_bot' (S₁₁)",
                    "pos_str": pos_s,
                    "freq_str": freq_str,
                    "frequencies": freqs,
                    "h_bounds": (cx, cx),
                    "v_bounds": v_b,
                }
            )
        elif pt.name == "in_top":
            if quadrature:
                x_src = cx - 0.12e-6
                p_quad = PhasedGaussianSource(pulse_obj, phase=-0.5 * np.pi)
                sources_s.append(
                    {
                        "kind": "sheet",
                        "type_name": "ModeSource",
                        "short_word": "TE₀ (−π/2)",
                        "desc": f"in_top Phased ModeSource (TE₀, φ = −π/2, {pos_s})",
                        "label": "in_top ModeSource (TE₀, φ = −π/2)",
                        "pos_str": f"x = {x_src * 1e6:.2f} µm, y = {cy * 1e6:.2f} µm",
                        "polarization": "TE₀ (E_y, H_z, φ = −π/2)",
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
                            "src_time": p_quad,
                        },
                    }
                )
            monitors_s.append(
                {
                    "kind": "plane",
                    "type_name": "ModeMonitor",
                    "short_word": "in_top",
                    "desc": f"in_top ModeMonitor (S₂₁, {pos_s})",
                    "label": "Port 2 'in_top' (S₂₁)",
                    "pos_str": pos_s,
                    "freq_str": freq_str,
                    "frequencies": freqs,
                    "h_bounds": (cx, cx),
                    "v_bounds": v_b,
                }
            )
        elif pt.name == "out_top":
            monitors_s.append(
                {
                    "kind": "plane",
                    "type_name": "ModeMonitor",
                    "short_word": "out_top",
                    "desc": f"out_top Cross ModeMonitor (S₃₁, {pos_s})",
                    "label": "Port 3 'out_top' (Cross S₃₁)",
                    "pos_str": pos_s,
                    "freq_str": freq_str,
                    "frequencies": freqs,
                    "h_bounds": (cx, cx),
                    "v_bounds": v_b,
                }
            )
        elif pt.name == "out_bot":
            monitors_s.append(
                {
                    "kind": "plane",
                    "type_name": "ModeMonitor",
                    "short_word": "out_bot",
                    "desc": f"out_bot Bar ModeMonitor (S₄₁, {pos_s})",
                    "label": "Port 4 'out_bot' (Bar S₄₁)",
                    "pos_str": pos_s,
                    "freq_str": freq_str,
                    "frequencies": freqs,
                    "h_bounds": (cx, cx),
                    "v_bounds": v_b,
                }
            )

    if include_flux_box:
        hb = (xmin + 2.0 * dx_bg, xmax - 2.0 * dx_bg)
        vb = (ymin + 2.0 * dy_bg, ymax - 2.0 * dy_bg)
        monitors_s.append(
            {
                "kind": "box",
                "type_name": "FluxMonitor",
                "short_word": "flux box",
                "desc": "4-wall Poynting FluxMonitor box",
                "label": "4-wall Poynting FluxMonitor box",
                "pos_str": f"x ∈ [{hb[0]*1e6:.2f}, {hb[1]*1e6:.2f}] µm, y ∈ [{vb[0]*1e6:.2f}, {vb[1]*1e6:.2f}] µm",
                "freq_str": freq_str,
                "frequencies": freqs,
                "h_bounds": hb,
                "v_bounds": vb,
            }
        )
    return sources_s, monitors_s


def save_coupler_setup_figures(
    pic,
    primary_data: dict,
    fig_dir,
    wavelength: float = 1.53e-6,
    min_feature: float = 207.02e-9,
):
    """
    Generate the setup figures and Markdown setup table for Tutorial 5.8 using
    ``wakis.schematic.save_setup_figures``:
    - ``00a_setup_geometry.png``: Single-port excitation setup schematic with
      Si/Air/SiO2 indices, port planes, launched ``TE_0`` mode profile,
      4-wall ``FluxMonitor`` box, and 10-cell CPML boundaries.
    - ``00b_setup_mesh_and_excitation.png``: 3-panel discretization view
      (computational mesh + 1× zoom inset at the ``g_eff = 207.0 nm`` coupling
      gap, spatial wave packet, and ``GaussianSource`` pulse/spectrum).
    - ``00c_quadrature_setup_schematic.png``: Companion setup schematic for the
      dual-port ``Delta phi = -pi/2`` quadrature excitation run (``task_9``).
    - ``00_setup_table.md``: Complete Markdown setup table.
    """
    from pathlib import Path

    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    neff_s = float(pic.n_eff_slab)
    eps_s = neff_s**2
    eps_clad = float(n_to_eps(pic.n_clad))
    proxy_solver = _SchematicSolverProxy(pic.solver, eps_clad)
    poly_structs = [
        _PolyStructProxy(
            s.vertices,
            eps_s,
            f"Si EIM core (n_Si=3.48, h=220 nm → n_eff,slab={neff_s:.4f}, w_eff=343 nm)",
        )
        for s in pic.structures_3d
    ]
    mat_names = {
        "background": f"Air lateral/upper clad (n={pic.n_clad:.2f}, ε_r={eps_clad:.2f}; SiO₂ BOX n_sub=1.45)",
    }
    sources_s, monitors_s = _build_coupler_schematic_io(
        pic, quadrature=False, include_flux_box=True
    )
    variations = [
        {
            "name": "Dual-port quadrature routing (task_9)",
            "details": "Simultaneous ModeSource on in_bot (φ = 0) and in_top (φ = −π/2) via PhasedGaussianSource",
        },
        {
            "name": "Coupling-length sweep (task_10..task_13)",
            "details": "L_c ∈ {0, 2, 4, 6, 8} µm at θ_sw = 30°",
        },
        {
            "name": "Sidewall-angle sweep (task_14..task_19)",
            "details": "θ_sw ∈ {0°, 5°, 10°, 15°, 20°, 25°, 30°} (g_eff ∈ [80.0, 207.0] nm)",
        },
        {
            "name": "Graded non-uniform Cartesian mesh (task_20)",
            "details": "Mirror-symmetric graded Δy (10.00 cells across g_eff) & non-square Δx > Δy_min (119,637 cells vs 193,842 at L3, −38.3%)",
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
        title_geometry="SOI 2×2 Directional Coupler Setup (2D EIM, Lc = 4.0 µm, θsw = 30°, geff = 207 nm)",
        title_mesh="Directional Coupler Discretization, Wave Packet & Broadband Excitation",
    )

    # Enhance 00a_setup_geometry.png with launched/cross TE0 mode profiles and coupling-gap callout
    fig_a, ax_a = schematic.plot_schematic(
        proxy_solver,
        plane="xy",
        structures=poly_structs,
        material_names=mat_names,
        sources=sources_s,
        monitors=monitors_s,
        unit="µm",
        wavelength=wavelength,
        title="SOI 2×2 Directional Coupler Setup (2D EIM, Lc = 4.0 µm, θsw = 30°, geff = 207 nm)",
    )
    if "port_y_in" in primary_data and "port_ey_in" in primary_data:
        yin = np.asarray(primary_data["port_y_in"], dtype=float) * 1e6
        eyin = np.abs(np.asarray(primary_data["port_ey_in"]))
        eyin = eyin / max(float(np.max(eyin)), 1e-30)
        x_in_um = float(pic.port_map["in_bot"].center[0]) * 1e6
        ax_a.fill_betweenx(
            yin, x_in_um, x_in_um + 1.05 * eyin, color="#D55E00", alpha=0.25, zorder=7
        )
        ax_a.plot(
            x_in_um + 1.05 * eyin,
            yin,
            color="#D55E00",
            lw=1.7,
            zorder=8,
            label="Launched TE₀ |E_y(y)|",
        )
    if "port_y_out" in primary_data and "port_ey_out" in primary_data:
        yout = np.asarray(primary_data["port_y_out"], dtype=float) * 1e6
        eyout = np.abs(np.asarray(primary_data["port_ey_out"]))
        eyout = eyout / max(float(np.max(eyout)), 1e-30)
        x_out_um = float(pic.port_map["out_top"].center[0]) * 1e6
        ax_a.fill_betweenx(
            yout, x_out_um - 1.05 * eyout, x_out_um, color="#009E73", alpha=0.22, zorder=7
        )
        ax_a.plot(
            x_out_um - 1.05 * eyout,
            yout,
            color="#009E73",
            lw=1.6,
            zorder=8,
        )
    figstyle.save(fig_a, res["geometry_figure"], close=True, auto_legend_below=False)

    # Enhance 00b_setup_mesh_and_excitation.png so PolySlab arms appear on ax_mesh and zoom targets the coupling gap
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
        zoom_bounds=(-0.45e-6, 0.45e-6, -0.48e-6, 0.48e-6),
        title="Directional Coupler Discretization, Wave Packet & Broadband Excitation",
    )
    ax_mesh = axes_b[0]
    for s in pic.structures_3d:
        ax_mesh.add_patch(
            MplPolygon(
                s.vertices * 1e6,
                closed=True,
                facecolor="#3B82F6",
                edgecolor="#1E293B",
                lw=1.0,
                alpha=0.45,
                zorder=2.5,
            )
        )
    figstyle.save(fig_b, res["mesh_figure"], close=True, auto_legend_below=False)

    # Render companion schematic 00c for the dual-port quadrature excitation setup (Rule 6)
    sources_q, monitors_q = _build_coupler_schematic_io(
        pic, quadrature=True, include_flux_box=False
    )
    fig_c, ax_c = schematic.plot_schematic(
        proxy_solver,
        plane="xy",
        structures=poly_structs,
        material_names=mat_names,
        sources=sources_q,
        monitors=monitors_q,
        unit="µm",
        wavelength=wavelength,
        title="Dual-Port Quadrature Excitation Setup (in_bot φ = 0, in_top φ = −π/2 → 99.65% Upper-Arm Routing)",
    )
    if "port_y_in" in primary_data and "port_ey_in" in primary_data:
        yin = np.asarray(primary_data["port_y_in"], dtype=float) * 1e6
        eyin = np.abs(np.asarray(primary_data["port_ey_in"]))
        eyin = eyin / max(float(np.max(eyin)), 1e-30)
        x_in_um = float(pic.port_map["in_bot"].center[0]) * 1e6
        for sgn_p, col_p in ((-1.0, "#D55E00"), (+1.0, "#7570b3")):
            y_arm = sgn_p * (-yin)
            ax_c.fill_betweenx(
                y_arm, x_in_um, x_in_um + 0.95 * eyin, color=col_p, alpha=0.24, zorder=7
            )
            ax_c.plot(x_in_um + 0.95 * eyin, y_arm, color=col_p, lw=1.6, zorder=8)
    path_c = fig_dir / "00c_quadrature_setup_schematic.png"
    figstyle.save(fig_c, path_c, close=True, auto_legend_below=False)
    res["quadrature_figure"] = path_c
    return res


def plot_mesh_upgrade_comparison_coupler(
    pic_uniform,
    pic_nonuniform,
    td_uniform: dict,
    td_nonuniform: dict,
    td_ref: dict | None,
    geom_base: dict | None = None,
    out_path=None,
):
    """
    Render ``04b_mesh_upgrade_comparison.png`` (3 panels <= 3) comparing the
    baseline ``kottke_L3`` mesh against the mirror-symmetric graded non-uniform
    Cartesian mesh ``kottke_nonuniform`` (``NonUniformCouplerPIC``):
    - Panel (a): Split-half mesh view (``y >= 0``: ``kottke_L3`` baseline,
      ``y <= 0``: ``kottke_nonuniform`` graded non-square mesh).
    - Panel (b): 1D transverse ``Δy(y)`` and axial ``Δx(x)`` cell-spacing
      profiles [nm] showing ``20.7 nm`` (10 cells) across the ``207 nm`` gap,
      ``51 nm`` in the outer air cladding, and non-square ``Δx ∈ [26, 38] nm``.
    - Panel (c): Broadband cross/bar transmittance ``T_cross(λ), T_bar(λ)``
      comparing ``kottke_L3``, ``kottke_nonuniform``, and ``kottke_reference``.
    """
    figstyle.apply_style()
    fig, (ax0, ax1, ax2) = figstyle.new_figure(nrows=1, ncols=3, width=15.4, height=4.7)

    # Panel (a): Split-half mesh view (y >= 0: kottke_L3, y <= 0: kottke_nonuniform)
    iz = pic_nonuniform.solver.Nz // 2
    ieps = np.asarray(pic_nonuniform.solver.ieps[:, :, iz, "y"])
    eps_r = np.where(ieps > 0, 1.0 / (np.where(ieps > 0, ieps, 1.0) * epsilon_0), 1.0)
    xu_nu = pic_nonuniform.grid.x * 1e6
    yu_nu = pic_nonuniform.grid.y * 1e6
    ax0.pcolormesh(
        xu_nu, yu_nu, eps_r.T, cmap=figstyle.CMAP_EPS, shading="flat", alpha=0.88
    )

    xl_m, yl_m = (-6.50, 6.50), (-2.95, 2.95)
    xu_u = pic_uniform.grid.x * 1e6
    yu_u = pic_uniform.grid.y * 1e6
    xv_u = xu_u[(xu_u >= xl_m[0]) & (xu_u <= xl_m[1])][::3]
    yv_u = yu_u[(yu_u >= 0.0) & (yu_u <= yl_m[1])][::2]
    segs_u = [[(x, 0.0), (x, yl_m[1])] for x in xv_u] + [
        [(xl_m[0], y), (xl_m[1], y)] for y in yv_u
    ]
    ax0.add_collection(
        LineCollection(segs_u, colors="#08519c", linewidths=0.38, alpha=0.65)
    )

    xv_nu = xu_nu[(xu_nu >= xl_m[0]) & (xu_nu <= xl_m[1])][::3]
    yv_nu = yu_nu[(yu_nu >= yl_m[0]) & (yu_nu <= 0.0)][::2]
    segs_nu = [[(x, yl_m[0]), (x, 0.0)] for x in xv_nu] + [
        [(xl_m[0], y), (xl_m[1], y)] for y in yv_nu
    ]
    ax0.add_collection(
        LineCollection(segs_nu, colors="#d95f02", linewidths=0.45, alpha=0.80)
    )
    ax0.axhline(0.0, color="black", lw=1.1, ls="--")
    for s in pic_nonuniform.structures_3d:
        ax0.add_patch(
            MplPolygon(
                s.vertices * 1e6,
                closed=True,
                fill=False,
                edgecolor="black",
                lw=1.1,
            )
        )

    n_u = int(td_uniform.get("n_cells", pic_uniform.grid.Nx * pic_uniform.grid.Ny))
    n_nu = int(td_nonuniform.get("n_cells", pic_nonuniform.grid.Nx * pic_nonuniform.grid.Ny))
    t_u_s = float(td_uniform.get("elapsed_s", 346.5))
    t_nu_s = float(td_nonuniform.get("elapsed_s", 162.9))
    red_cells = 100.0 * (1.0 - n_nu / max(n_u, 1))
    red_time = 100.0 * (1.0 - t_nu_s / t_u_s) if t_u_s > 0 else 0.0

    ax0.text(
        -6.25,
        2.38,
        f"Upper (y > 0): kottke_L3 baseline\n"
        f"{pic_uniform.grid.Nx}×{pic_uniform.grid.Ny} = {n_u:,} cells ({t_u_s:.1f} s)",
        fontsize=7.3,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#08519c", alpha=0.92),
    )
    ax0.text(
        -6.25,
        -2.72,
        f"Lower (y < 0): kottke_nonuniform (graded)\n"
        f"{pic_nonuniform.grid.Nx}×{pic_nonuniform.grid.Ny} = {n_nu:,} cells "
        f"(−{red_cells:.1f}%, {t_nu_s:.1f} s, −{red_time:.1f}%)",
        fontsize=7.3,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#d95f02", alpha=0.92),
    )
    ax0.set(
        xlim=xl_m,
        ylim=yl_m,
        aspect="auto",
        xlabel="x [µm]",
        ylabel="y [µm]",
        title="(a) Baseline L3 (top) vs graded non-uniform mesh (bottom)",
    )

    # Panel (b): 1D cell spacing profiles dy(y) and dx(x) [nm]
    yc_u = 0.5 * (yu_u[:-1] + yu_u[1:])
    dy_u = np.diff(yu_u) * 1e3
    yc_nu = 0.5 * (yu_nu[:-1] + yu_nu[1:])
    dy_nu = np.diff(yu_nu) * 1e3
    xc_nu = 0.5 * (xu_nu[:-1] + xu_nu[1:])
    dx_nu = np.diff(xu_nu) * 1e3
    ax1.axvspan(-0.1035, +0.1035, color="#fdae61", alpha=0.35, label="Slot geff = 207 nm (10 cells)")
    ax1.step(
        yc_u,
        dy_u,
        where="mid",
        color="#08519c",
        lw=1.7,
        label=f"kottke_L3 Δy(y) ({pic_uniform.grid.Ny} cells)",
    )
    ax1.step(
        yc_nu,
        dy_nu,
        where="mid",
        color="#d95f02",
        lw=2.0,
        label=f"kottke_nonuniform Δy(y) ({pic_nonuniform.grid.Ny} cells)",
    )
    ax1.step(
        xc_nu * (3.0 / 8.65),
        dx_nu,
        where="mid",
        color="#1b9e77",
        ls="--",
        lw=1.6,
        label=f"kottke_nonuniform Δx(x·Ly/Lx) ({pic_nonuniform.grid.Nx} cells)",
    )
    ax1.set(
        xlim=(-3.1, 3.1),
        ylim=(16.0, 56.0),
        xlabel="Transverse coordinate y [µm] (or scaled x)",
        ylabel="Cell spacing Δy, Δx [nm]",
        title="(b) Graded Δy(y) & non-square Δx(x) profiles",
    )
    ax1.grid(True, ls=":", alpha=0.4)
    ax1.legend(loc="upper center", fontsize=7.0)

    # Panel (c): Broadband cross/bar transmittance comparison
    def _spec_c(d):
        w = np.asarray(d["wavelengths"], dtype=float) * 1e9
        o = np.argsort(w)
        tc = np.abs(np.asarray(d["S31"])[o]) ** 2
        tb = np.abs(np.asarray(d["S41"])[o]) ** 2
        wlc = float(d["wl_cross"]) * 1e9
        return w[o], tc, tb, wlc

    w_u, tc_u, tb_u, wlc_u = _spec_c(td_uniform)
    w_nu, tc_nu, tb_nu, wlc_nu = _spec_c(td_nonuniform)
    if td_ref is not None:
        w_r, tc_r, tb_r, wlc_r = _spec_c(td_ref)
        ax2.plot(
            w_r,
            tc_r,
            color="#111111",
            lw=2.1,
            label=f"Ref R Cross (λc={wlc_r:.2f} nm)",
        )
        ax2.plot(w_r, tb_r, color="#111111", lw=1.5, ls=":", alpha=0.75)
    ax2.plot(
        w_u,
        tc_u,
        "--",
        color="#08519c",
        lw=1.9,
        label=f"kottke_L3 Cross ({n_u:,}c, λc={wlc_u:.2f} nm)",
    )
    ax2.plot(w_u, tb_u, "--", color="#08519c", lw=1.3, alpha=0.65)
    ax2.plot(
        w_nu,
        tc_nu,
        "o-",
        color="#d95f02",
        ms=3.8,
        lw=1.8,
        label=f"kottke_nonuniform Cross ({n_nu:,}c, λc={wlc_nu:.2f} nm)",
    )
    ax2.plot(w_nu, tb_nu, "s-", color="#d95f02", ms=3.2, lw=1.3, alpha=0.75)
    ax2.axhline(0.5, color="0.45", ls=":", lw=1.1)
    ax2.set(
        xlabel="Wavelength λ [nm]",
        ylabel="Guided power fraction (Cross / Bar)",
        ylim=(-0.03, 1.05),
        title="(c) Equal-accuracy cross/bar splitting spectra",
    )
    ax2.grid(True, ls=":", alpha=0.4)
    ax2.legend(loc="center left", fontsize=7.0)

    figstyle.save(fig, out_path, close=True, auto_legend_below=False)
    return fig


def overlay_coupler_geometry(
    ax,
    pic,
    outline_color: str = "white",
    quadrature: bool = False,
):
    """Overlay material outlines/labels, port planes, and CPML boundaries on a 2D coupler field map."""
    neff_s = float(pic.n_eff_slab)
    eps_clad = float(n_to_eps(pic.n_clad))
    sources_s, monitors_s = _build_coupler_schematic_io(
        pic, quadrature=quadrature, include_flux_box=False
    )
    schematic.overlay_geometry(
        ax,
        _SchematicSolverProxy(pic.solver, eps_clad),
        plane="xy",
        unit="µm",
        structures=(),
        material_names={"background": f"Air clad (n={pic.n_clad:.2f})"},
        sources=sources_s,
        monitors=monitors_s,
        show_materials=False,
        label_materials=False,
        outline_color=outline_color,
    )
    for s in pic.structures_3d:
        ax.add_patch(
            MplPolygon(
                s.vertices * 1e6,
                closed=True,
                fill=False,
                edgecolor=outline_color,
                lw=1.15,
                alpha=0.92,
                zorder=6,
            )
        )
    ax.text(
        0.0,
        -2.52,
        f"Si EIM core (n_eff,slab = {neff_s:.2f}, w_eff = 343 nm, g_eff = 207 nm) | Air clad (n = 1.00)",
        fontsize=7.3,
        fontweight="bold",
        color="#0f172a",
        ha="center",
        va="center",
        zorder=9,
        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="#64748b", alpha=0.88, lw=0.8),
    )


def render_coupler_field_map_with_overlay(
    x: np.ndarray,
    y: np.ndarray,
    field: np.ndarray,
    pic,
    kind: str,
    title: str,
    out_path,
    quadrature: bool = False,
):
    """Render a 2D directional-coupler steady-state field map with ``figstyle`` and ``overlay_geometry``."""
    figstyle.apply_style()
    fig, ax = figstyle.new_figure(nrows=1, ncols=1, width=9.4, height=4.4)
    x_um = np.asarray(x) * 1e6
    y_um = np.asarray(y) * 1e6
    if kind == "intensity_log":
        db_floor = -35.0
        f_abs = np.asarray(field, dtype=float)
        data_db = 10.0 * np.log10(
            np.maximum(
                f_abs / (float(np.max(f_abs)) or 1.0), 10.0 ** (db_floor / 10.0)
            )
        )
        im = ax.pcolormesh(
            x_um,
            y_um,
            data_db.T,
            cmap="inferno",
            vmin=db_floor,
            vmax=0.0,
            shading="auto",
        )
        cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        cb.set_label(r"$10\log_{10}(|\mathbf{E}|^2 / |\mathbf{E}|_{\max}^2)$ [dB]")
        outline_col = "white"
    else:
        f_re = np.real(field)
        vmax = float(np.percentile(np.abs(f_re), 99.5)) or 1.0
        im = ax.pcolormesh(
            x_um,
            y_um,
            (f_re / vmax).T,
            cmap=figstyle.CMAP_DIVERGING,
            vmin=-1.0,
            vmax=1.0,
            shading="auto",
        )
        cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        cb.set_label(r"Normalized $\mathrm{Re}(E_y)$")
        outline_col = "black"

    overlay_coupler_geometry(ax, pic, outline_color=outline_col, quadrature=quadrature)
    ax.set(
        xlim=(float(x_um.min()), float(x_um.max())),
        ylim=(float(y_um.min()), float(y_um.max())),
        aspect="equal",
        xlabel="x [µm]",
        ylabel="y [µm]",
        title=title,
    )
    figstyle.save(fig, out_path, close=True, auto_legend_below=False)


def render_coupler_movie_with_overlay(
    x: np.ndarray,
    y: np.ndarray,
    frames: np.ndarray,
    times: np.ndarray,
    pic,
    out_path,
    fps: int = 12,
):
    """Render animated GIF ``09_pulse_propagation.gif`` (< 5 MB) with ``overlay_geometry`` on every frame."""
    from matplotlib.animation import FuncAnimation, PillowWriter

    figstyle.apply_style()
    fig, ax = plt.subplots(figsize=(6.8, 3.2), dpi=84)
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
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label(r"$E_y(x, y, t) / E_{\max}$")
    overlay_coupler_geometry(ax, pic, outline_color="black", quadrature=False)
    ax.set(
        xlim=(float(x_um.min()), float(x_um.max())),
        ylim=(float(y_um.min()), float(y_um.max())),
        aspect="equal",
        xlabel="x [µm]",
        ylabel="y [µm]",
    )
    ttl = ax.set_title(
        f"Directional coupler pulse propagation (t = {float(times[0])*1e15:.1f} fs)"
    )
    fig.tight_layout()

    def _update(frame_idx):
        im.set_array((f_arr[frame_idx] / vmax).T.ravel())
        ttl.set_text(
            f"Directional coupler pulse propagation (t = {float(times[frame_idx])*1e15:.1f} fs)"
        )
        return (im, ttl)

    anim = FuncAnimation(
        fig, _update, frames=len(f_arr), interval=1000 // fps, blit=False
    )
    anim.save(str(out_path), writer=PillowWriter(fps=fps))
    plt.close(fig)


if __name__ == "__main__":
    from directional_coupler import main

    main()

