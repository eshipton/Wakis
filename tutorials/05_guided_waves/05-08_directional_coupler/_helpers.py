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
import numpy as np
from scipy.constants import c as c_light

from wakis.field_monitors import DFTMonitor
from wakis.mode_solver import ModeMonitor, ModeSource, solve_modes
from wakis.templates.photonics import n_to_eps, slab_neff, vertical_slab_mode
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

    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.8), dpi=150)
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

    fig.tight_layout()
    if out_path is not None:
        from pathlib import Path

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig, axes


if __name__ == "__main__":
    from directional_coupler import main

    main()

