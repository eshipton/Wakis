# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""
Physics simulation runners and plotting helpers for Tutorial 8.1:
Space-Charge Waves on a Drifting Electron Beam (`wakis.beam_fluid.LinearizedBeam`).
"""

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from PIL import Image
from scipy.constants import c as c_light, e as q_e, epsilon_0 as eps_0, m_e
from scipy.optimize import brentq
from scipy.special import jn as bessel_jn

import wakis
from wakis import figstyle, schematic
from wakis.beam_fluid import (
    LinearizedBeam,
    ballistic_bunching_analytic,
    beam_plasma_frequency,
    reduction_factor_cylindrical,
    reduction_factor_planar,
    reduction_factor_rectangular,
    relativistic_gamma,
    space_charge_wave_dispersion,
)

# ---------------------------------------------------------------------------
# Benchmark parameters from Mealy, Marosi & Capolino (2022) & Branch-Mihran (1955)
# ---------------------------------------------------------------------------
# Reference: T. Mealy, R. Marosi, and F. Capolino, IEEE Trans. Plasma Sci. 50(10),
# 3570-3577 (2022), DOI: 10.1109/TPS.2022.3201036 (arXiv:2205.05806)
U0_REF = 0.16 * c_light  # 4.7967e7 m/s (V0 = 6.70 kV, gamma0 = 1.01305)
_, GAMMA0_REF = relativistic_gamma(u0=U0_REF)
V0_REF = (GAMMA0_REF - 1.0) * m_e * c_light**2 / q_e  # ~6.70 kV

A_TUNNEL_REF = 2.0e-3  # 2.0 mm tunnel half-height / radius
B_BEAM_REF = 0.5e-3  # 0.5 mm beam half-thickness / radius (b/a = 0.25)
F0_REF = 5.0e9  # 5.0 GHz nominal modulation frequency
OMEGA0_REF = 2.0 * np.pi * F0_REF
BETA0_REF = OMEGA0_REF / U0_REF  # 654.95 rad/m (lambda_e = 9.593 mm)
LAMBDA_E_REF = U0_REF / F0_REF  # 9.593 mm electronic wavelength

# Plasma frequency used for crisp multi-period beat resolution in compact domains
# (In Branch-Mihran slow-wave theory, R(omega, u0, b, a) is independent of omega_p)
FP_REF = 1.35e9  # 1.35 GHz plasma frequency
WP_REF = 2.0 * np.pi * FP_REF

# Velocity-modulating grid gap parameters (Mealy et al. 2022, Section III)
D_GRID_REF = 0.2e-3  # 0.2 mm grid gap
VM_REF = 100.0  # 100 V small-signal modulation voltage (Vm / V0 = 0.0149)


def planar_tm_eigenmode(y, omega, u0, b_half, a_half):
    """
    Exact transverse TM eigenmode profiles ``(Es(y), Eperp(y), J1s(y), T, tau, R)``
    for a symmetric 2D planar sheet beam of half-thickness ``b_half`` inside a
    parallel-plate PEC drift tunnel of half-height ``a_half`` (Branch & Mihran 1955).

    Inside the beam (``|y| <= b_half``)::

        Es(y) = cos(T y),
        Eperp(y) = (tau / T) sin(T y),

    Outside the beam (``b_half < |y| <= a_half``)::

        Es(y) = cos(T b) * sinh(tau (a - |y|)) / sinh(tau (a - b)),
        Eperp(y) = sign(y) * cos(T b) * cosh(tau (a - |y|)) / sinh(tau (a - b)).
    """
    y = np.asarray(y, dtype=float)
    _, gamma0 = relativistic_gamma(u0=u0)
    beta_e = abs(float(omega)) / abs(float(u0))
    tau = beta_e / gamma0
    b = float(b_half)
    a = float(a_half)
    if np.isclose(b, a):
        T = 0.5 * np.pi / b
    else:
        rhs = tau * b / np.tanh(tau * (a - b))
        u_root = brentq(
            lambda u: u * np.tan(u) - rhs, 0.0, 0.5 * np.pi * (1.0 - 1e-12)
        )
        T = u_root / b
    R = 1.0 / np.sqrt(1.0 + (T / tau) ** 2)

    inside = np.abs(y) <= b + 1e-12
    Es = np.zeros_like(y)
    Eperp = np.zeros_like(y)
    Es[inside] = np.cos(T * y[inside])
    Eperp[inside] = (tau / max(T, 1e-30)) * np.sin(T * y[inside])

    if not np.isclose(b, a):
        out = ~inside
        sh = np.sinh(tau * (a - b))
        Es[out] = np.cos(T * b) * np.sinh(tau * (a - np.abs(y[out]))) / sh
        Eperp[out] = (
            np.sign(y[out])
            * np.cos(T * b)
            * np.cosh(tau * (a - np.abs(y[out])))
            / sh
        )
    J1s = np.where(inside, Es, 0.0)
    return Es, Eperp, J1s, float(T), float(tau), float(R)


def extract_space_charge_doublet(trace, dt, f0_expected, wp_expected):
    """
    Extract fast and slow space-charge wave angular frequencies ``(omega_fast,
    omega_slow, omega_q)`` and normalized FFT spectrum from a complex Bloch
    time series ``trace(t_n)``.

    Uses a Hann-windowed zero-padded FFT with 3-point Gaussian (log-parabolic)
    sub-bin peak interpolation in the passband around ``f0_expected``.
    """
    nt = len(trace)
    win = np.hanning(nt)
    n_fft = max(131072, 1 << int(np.ceil(np.log2(nt * 16))))
    spec = np.abs(np.fft.fft(trace * win, n=n_fft))
    freqs = np.fft.fftfreq(n_fft, d=dt)

    pos = freqs > 0
    f_pos = freqs[pos]
    s_pos = spec[pos]

    fp = wp_expected / (2.0 * np.pi)
    half_bw = max(2.2 * fp, 0.45 * f0_expected)
    band = (f_pos > max(0.05 * f0_expected, f0_expected - half_bw)) & (
        f_pos < f0_expected + half_bw
    )
    f_band = f_pos[band]
    s_band = s_pos[band]

    def _interp_peak(mask_sub):
        idx_sub = np.where(mask_sub)[0]
        k = int(idx_sub[np.argmax(s_band[mask_sub])])
        if 0 < k < len(s_band) - 1:
            y0, y1, y2 = np.log(s_band[k - 1 : k + 2] + 1e-30)
            denom = y0 - 2.0 * y1 + y2
            if abs(denom) > 1e-30:
                delta = 0.5 * (y0 - y2) / denom
                df = f_band[1] - f_band[0]
                return float(f_band[k] + delta * df)
        return float(f_band[k])

    f_slow = _interp_peak(f_band < f0_expected)
    f_fast = _interp_peak(f_band > f0_expected)
    omega_slow = 2.0 * np.pi * f_slow
    omega_fast = 2.0 * np.pi * f_fast
    omega_q = 0.5 * (omega_fast - omega_slow)

    spec_norm = s_band / max(float(np.max(s_band)), 1e-30)
    return omega_fast, omega_slow, omega_q, f_band, spec_norm


def run_2d_bloch_dispersion_point(
    b_half=B_BEAM_REF,
    a_half=A_TUNNEL_REF,
    f0=F0_REF,
    u0=U0_REF,
    wp=WP_REF,
    nx=24,
    ny=40,
    n_beat_periods=1.8,
    cfln=0.75,
    gamma_power=3.0,
    coupling_sign=1.0,
    record_profiles=False,
):
    """
    Measure 2D planar sheet-beam space-charge wave dispersion ``(omega_+, omega_-,
    omega_q, R)`` in a parallel-plate PEC drift tunnel using ``SolverFIT3D`` in
    2D mode (``Nz=1``, ``axis='x'``, Bloch boundary along ``x``, PEC along ``y``).
    """
    omega0 = 2.0 * np.pi * float(f0)
    beta = omega0 / float(u0)
    Lx = 2.0 * np.pi / beta
    dx = Lx / int(nx)
    dy = (2.0 * float(a_half)) / int(ny)

    grid = wakis.GridFIT3D(
        0.0,
        Lx,
        -float(a_half),
        float(a_half),
        -0.5 * dx,
        0.5 * dx,
        int(nx),
        int(ny),
        1,
        verbose=0,
    )
    solver = wakis.SolverFIT3D(
        grid,
        bc_low=["bloch", "pec", "periodic"],
        bc_high=["bloch", "pec", "periodic"],
        k_bloch=[-beta, 0.0, 0.0],
        dtype=np.complex128,
        cfln=cfln,
        verbose=0,
    )
    _, gamma0 = relativistic_gamma(u0=u0)
    beam = LinearizedBeam(
        u0=u0,
        omega_p=wp,
        axis="x",
        half_thickness=b_half,
        gamma_power=gamma_power,
        coupling_sign=coupling_sign,
        solver=solver,
    )

    # Exact transverse profile for clean fundamental-mode excitation
    y_nodes = grid.y[: solver.Ny]
    Es_prof, Ey_prof, J_prof, T_th, tau_th, R_th = planar_tm_eigenmode(
        y_nodes, omega0, u0, b_half, a_half
    )
    phase = np.exp(-1j * beta * beam.s_coords)[:, None]
    beam.v1[:] = 1.0e4 * phase * Es_prof[beam.keep_lines][None, :]
    solver.one_step()

    omega_q_th = R_th * (wp / (gamma0 ** (0.5 * gamma_power)))
    t_beat = 2.0 * np.pi / max(omega_q_th, 0.1 * wp)
    nt = max(int(np.ceil(n_beat_periods * t_beat / solver.dt)), 1600)

    trace_Ex = np.zeros(nt, dtype=np.complex128)
    iy_mid = solver.Ny // 2
    ix_probe = int(nx) // 4

    # Accumulate transverse mode profiles near peak E-field if requested
    n_snap = int(0.25 * (np.pi / max(omega_q_th, 1e-12)) / solver.dt)
    n_snap = min(max(n_snap, 50), nt - 1)
    Ex_slice = None
    Ey_slice = None
    J1_slice = None

    for n in range(nt):
        solver.one_step()
        trace_Ex[n] = solver.E[ix_probe, iy_mid, 0, "x"]
        if record_profiles and n == n_snap:
            # Demodulate axial Bloch factor exp(-i beta x)
            Ex_slice = np.real(
                solver.E[ix_probe, : solver.Ny, 0, "x"]
                * np.exp(1j * beta * beam.s_coords[ix_probe])
            )
            Ey_raw = solver.E[ix_probe, : solver.Ny - 1, 0, "y"] * np.exp(
                1j * beta * beam.s_coords[ix_probe]
            )
            # Ey is 90 deg out of phase with Ex in a TM standing-envelope mode
            Ey_slice = np.imag(Ey_raw)
            j_3d = beam.current_density_field()
            J1_slice = np.real(
                j_3d[ix_probe, : solver.Ny, 0]
                * np.exp(1j * beta * beam.s_coords[ix_probe])
            )

    omega_fast, omega_slow, omega_q_meas, f_band, spec_norm = (
        extract_space_charge_doublet(trace_Ex, solver.dt, f0, wp)
    )
    R_meas = omega_q_meas * (gamma0 ** (0.5 * gamma_power)) / wp
    omega_fast_th, omega_slow_th = space_charge_wave_dispersion(
        beta, u0, wp, R=R_th
    )

    min_feature = float(b_half)
    dx_bg = float(max(dx, dy))
    dx_min = float(min(dx, dy))
    cells_across = float(b_half) / dy
    cpw = (float(u0) / float(f0)) / dx

    out = {
        "f0": float(f0),
        "beta": float(beta),
        "u0": float(u0),
        "gamma0": float(gamma0),
        "wp": float(wp),
        "b_half": float(b_half),
        "a_half": float(a_half),
        "omega_fast": float(omega_fast),
        "omega_slow": float(omega_slow),
        "omega_q": float(omega_q_meas),
        "R_fit": float(R_meas),
        "omega_fast_th": float(omega_fast_th),
        "omega_slow_th": float(omega_slow_th),
        "omega_q_th": float(0.5 * (omega_fast_th - omega_slow_th)),
        "R_th": float(R_th),
        "T_th": float(T_th),
        "tau_th": float(tau_th),
        "dt": float(solver.dt),
        "nt": int(nt),
        "nx": int(nx),
        "ny": int(ny),
        "nz": 1,
        "dx_background": dx_bg,
        "dx_min": dx_min,
        "min_feature": min_feature,
        "cells_across_feature": cells_across,
        "cpw": cpw,
        "n_cells": int(nx) * int(ny),
        "n_steps": int(nt),
        "trace_t": (np.arange(nt, dtype=np.float64) * solver.dt),
        "trace_Ex": trace_Ex,
        "fft_freqs": f_band.astype(np.float64),
        "fft_spec": spec_norm.astype(np.float64),
    }
    if record_profiles:
        y_edges = 0.5 * (y_nodes[:-1] + y_nodes[1:])
        _, Ey_th_edges, _, _, _, _ = planar_tm_eigenmode(
            y_edges, omega0, u0, b_half, a_half
        )
        out["y_nodes"] = y_nodes.astype(np.float64)
        out["y_edges"] = y_edges.astype(np.float64)
        out["Es_th"] = Es_prof.astype(np.float64)
        out["Ey_th"] = Ey_th_edges.astype(np.float64)
        out["J1_th"] = J_prof.astype(np.float64)
        out["Ex_fit"] = (
            Ex_slice / max(float(np.max(np.abs(Ex_slice))), 1e-30)
        ).astype(np.float64)
        # Match sign of Ey_fit to Ey_th
        ey_norm = Ey_slice / max(float(np.max(np.abs(Ey_slice))), 1e-30)
        if np.dot(ey_norm, Ey_th_edges) < 0:
            ey_norm = -ey_norm
        es_scale = float(np.max(np.abs(Ey_th_edges)))
        out["Ey_fit"] = (ey_norm * es_scale).astype(np.float64)
        j_norm = J1_slice / max(float(np.max(np.abs(J1_slice))), 1e-30)
        if np.dot(j_norm, J_prof) < 0:
            j_norm = -j_norm
        out["J1_fit"] = j_norm.astype(np.float64)
        w_perp_profile = np.zeros(solver.Ny, dtype=np.float64)
        w_perp_profile[beam.keep_lines] = (
            beam.rho0[0, :] / beam.rho0_peak
        )
        out["w_perp_y"] = w_perp_profile
    return out


def run_3d_bloch_dispersion_point(
    b_half=0.8e-3,
    a_half=A_TUNNEL_REF,
    f0=F0_REF,
    u0=U0_REF,
    wp=2.0 * np.pi * 1.65e9,
    nx=20,
    ny=20,
    nz=16,
    n_beat_periods=1.65,
    cfln=0.78,
):
    """
    Measure full 3D space-charge wave dispersion and transverse electric field
    fringing ``(Ex, Ey, Ez)`` for a drifting electron beam along ``z`` inside a
    square PEC drift tunnel ``[-a_half, a_half]^2``.
    """
    omega0 = 2.0 * np.pi * float(f0)
    beta = omega0 / float(u0)
    Lz = 2.0 * np.pi / beta
    dx = (2.0 * float(a_half)) / int(nx)
    dy = (2.0 * float(a_half)) / int(ny)
    dz = Lz / int(nz)

    grid = wakis.GridFIT3D(
        -float(a_half),
        float(a_half),
        -float(a_half),
        float(a_half),
        0.0,
        Lz,
        int(nx),
        int(ny),
        int(nz),
        verbose=0,
    )
    solver = wakis.SolverFIT3D(
        grid,
        bc_low=["pec", "pec", "bloch"],
        bc_high=["pec", "pec", "bloch"],
        k_bloch=[0.0, 0.0, -beta],
        dtype=np.complex128,
        cfln=cfln,
        verbose=0,
    )
    _, gamma0 = relativistic_gamma(u0=u0)
    beam = LinearizedBeam(
        u0=u0,
        omega_p=wp,
        axis="z",
        half_thickness=(b_half, b_half),
        solver=solver,
    )

    # Fundamental 2D transverse profile cos(Tx x) cos(Ty y) inside the beam
    x_nodes = grid.x[: solver.Nx]
    y_nodes = grid.y[: solver.Ny]
    Es_x, _, _, _, _, _ = planar_tm_eigenmode(x_nodes, omega0, u0, b_half, a_half)
    Es_y, _, _, _, _, _ = planar_tm_eigenmode(y_nodes, omega0, u0, b_half, a_half)
    prof_2d = np.outer(Es_x, Es_y).ravel()[beam.keep_lines]
    phase = np.exp(-1j * beta * beam.s_coords)[:, None]
    beam.v1[:] = 1.0e4 * phase * prof_2d[None, :]
    solver.one_step()

    R_3d_rect_th = reduction_factor_rectangular(
        omega0, u0, b_half, b_half, a_half, a_half
    )
    R_3d_cyl_th = reduction_factor_cylindrical(omega0, u0, b_half, a_half)
    R_2d_th = reduction_factor_planar(omega0, u0, b_half, a_half)

    omega_q_th = R_3d_rect_th * wp / (gamma0**1.5)
    t_beat = 2.0 * np.pi / max(omega_q_th, 0.1 * wp)
    nt = max(int(np.ceil(n_beat_periods * t_beat / solver.dt)), 1400)

    trace_Ez = np.zeros(nt, dtype=np.complex128)
    ix_mid = solver.Nx // 2
    iy_mid = solver.Ny // 2
    iz_probe = int(nz) // 4
    n_snap = min(max(int(0.25 * t_beat / solver.dt), 40), nt - 1)

    Ez_xy = None
    Ex_xy = None
    Ey_xy = None
    for n in range(nt):
        solver.one_step()
        trace_Ez[n] = solver.E[ix_mid, iy_mid, iz_probe, "z"]
        if n == n_snap:
            ph_z = np.exp(1j * beta * beam.s_coords[iz_probe])
            Ez_xy = np.real(
                solver.E[: solver.Nx, : solver.Ny, iz_probe, "z"] * ph_z
            )
            Ex_xy = np.imag(
                solver.E[: solver.Nx - 1, : solver.Ny, iz_probe, "x"] * ph_z
            )
            Ey_xy = np.imag(
                solver.E[: solver.Nx, : solver.Ny - 1, iz_probe, "y"] * ph_z
            )

    omega_fast, omega_slow, omega_q_meas, f_band, spec_norm = (
        extract_space_charge_doublet(trace_Ez, solver.dt, f0, wp)
    )
    R_3d_fit = omega_q_meas * (gamma0**1.5) / wp

    return {
        "f0": float(f0),
        "b_half": float(b_half),
        "a_half": float(a_half),
        "omega_fast": float(omega_fast),
        "omega_slow": float(omega_slow),
        "omega_q": float(omega_q_meas),
        "R_3d_fit": float(R_3d_fit),
        "R_3d_rect_th": float(R_3d_rect_th),
        "R_3d_cyl_th": float(R_3d_cyl_th),
        "R_2d_th": float(R_2d_th),
        "x_nodes": x_nodes.astype(np.float64),
        "y_nodes": y_nodes.astype(np.float64),
        "Ez_xy": Ez_xy.astype(np.float64),
        "Ex_xy": Ex_xy.astype(np.float64),
        "Ey_xy": Ey_xy.astype(np.float64),
        "fft_freqs": f_band.astype(np.float64),
        "fft_spec": spec_norm.astype(np.float64),
        "dx_background": float(max(dx, dy, dz)),
        "dx_min": float(min(dx, dy, dz)),
        "min_feature": float(b_half),
        "cells_across_feature": float(b_half) / max(dx, dy),
        "cpw": (float(u0) / float(f0)) / dz,
        "n_cells": int(nx) * int(ny) * int(nz),
        "n_steps": int(nt),
    }


def run_modulated_drift_tube(
    b_half=0.8e-3,
    a_half=A_TUNNEL_REF,
    L_tube=0.115,
    f_mod=F0_REF,
    u0=U0_REF,
    wp=2.0 * np.pi * 1.05e9,
    V_m=VM_REF,
    nx=240,
    ny=20,
    n_pml=6,
    ramp_cells=10,
    record_movie=True,
    n_frames=28,
):
    """
    Simulate klystron-like small-signal velocity modulation and space-charge wave
    standing-wave bunching along an open-boundary 2D drift tube (reproducing the
    physics of Mealy, Marosi & Capolino 2022, Fig. 1b).
    """
    dx = float(L_tube) / int(nx)
    dy = (2.0 * float(a_half)) / int(ny)
    grid = wakis.GridFIT3D(
        0.0,
        float(L_tube),
        -float(a_half),
        float(a_half),
        -0.5 * dx,
        0.5 * dx,
        int(nx),
        int(ny),
        1,
        verbose=0,
    )
    solver = wakis.SolverFIT3D(
        grid,
        bc_low=["cpml", "pec", "periodic"],
        bc_high=["cpml", "pec", "periodic"],
        n_pml=int(n_pml),
        cfln=0.5,
        verbose=0,
    )
    _, gamma0 = relativistic_gamma(u0=u0)
    V0 = (gamma0 - 1.0) * m_e * c_light**2 / q_e

    # Modulating gap placed 3 cells downstream of inflow plane, spanning 4 cells (>= 4 cells across feature)
    s_in = (int(n_pml) + 3) * dx
    d_gap = 4.0 * dx
    s_gap = s_in + 0.5 * d_gap
    sigma_gap = 0.5 * d_gap

    omega_mod = 2.0 * np.pi * float(f_mod)
    beta_0 = omega_mod / float(u0)
    theta_g = omega_mod * d_gap / float(u0)
    M_gap = float(np.sinc(theta_g / (2.0 * np.pi)))

    # Peak velocity kick dv_max corresponding to gap voltage V_m
    dv_max = (q_e * float(V_m) * M_gap) / (gamma0**3 * m_e * float(u0))
    t_rise = 2.0 / float(f_mod)

    def gap_kick(s_coords, t):
        env = np.tanh(t / t_rise) ** 2
        w_gap = np.exp(-0.5 * ((s_coords - s_gap) / sigma_gap) ** 2) / (
            np.sqrt(2.0 * np.pi) * sigma_gap
        )
        return (dv_max * float(u0)) * w_gap * env * np.sin(omega_mod * t)

    beam = LinearizedBeam(
        u0=u0,
        omega_p=wp,
        axis="x",
        half_thickness=b_half,
        ramp_cells=int(ramp_cells),
        gap_modulation=gap_kick,
        solver=solver,
    )

    # Transit time across the drift tube + 2 modulation periods for steady harmonicphasor
    t_transit = float(L_tube) / float(u0)
    t_total = 1.45 * t_transit
    t_meas = 2.0 / float(f_mod)
    nt_settle = int(np.ceil(t_total / solver.dt))
    nt_meas = int(np.ceil(t_meas / solver.dt))
    nt_total = nt_settle + nt_meas

    movie_frames_rho = []
    movie_frames_Ex = []
    movie_times = []
    frame_stride = max(1, nt_total // max(int(n_frames), 1))

    for n in range(nt_settle):
        solver.one_step()
        if record_movie and (n % frame_stride == 0) and len(movie_times) < n_frames:
            movie_times.append(float((n + 1) * solver.dt))
            movie_frames_rho.append(
                beam.charge_density_field()[:, :, 0].astype(np.float32)
            )
            movie_frames_Ex.append(
                solver.E[: solver.Nx, : solver.Ny, 0, "x"].astype(np.float32)
            )

    j_phasor = np.zeros(beam.M, dtype=np.complex128)
    v_phasor = np.zeros(beam.M, dtype=np.complex128)
    for m in range(nt_meas):
        solver.one_step()
        t_cur = (nt_settle + m + 0.5) * solver.dt
        ph = np.exp(-1j * omega_mod * t_cur)
        # Integrated AC beam current I_1(x) and cross-section-averaged kinetic voltage V_1(x)
        i_line = np.sum(beam.J1 * beam.dA_perp, axis=1)
        v_line = np.mean(beam.kinetic_voltage(), axis=1)
        j_phasor += i_line * ph
        v_phasor += v_line * ph
        n_glob = nt_settle + m
        if (
            record_movie
            and (n_glob % frame_stride == 0)
            and len(movie_times) < n_frames
        ):
            movie_times.append(float((n_glob + 1) * solver.dt))
            movie_frames_rho.append(
                beam.charge_density_field()[:, :, 0].astype(np.float32)
            )
            movie_frames_Ex.append(
                solver.E[: solver.Nx, : solver.Ny, 0, "x"].astype(np.float32)
            )

    j_phasor *= 2.0 / nt_meas
    v_phasor *= 2.0 / nt_meas

    # Active drift region downstream of the modulating gap and upstream of the outflow ramp
    s_rel = beam.s_coords - s_gap
    valid = (s_rel >= 1.5 * sigma_gap) & (
        np.arange(beam.M) < beam.M - int(ramp_cells) - 2
    )
    z_drift = s_rel[valid]
    I1_env = np.abs(j_phasor[valid])
    V1_env = np.abs(v_phasor[valid])

    R_th = reduction_factor_planar(omega_mod, u0, b_half, a=a_half)
    omega_q_th = R_th * wp / (gamma0**1.5)
    beta_q_th = omega_q_th / float(u0)
    L_beat_th = np.pi / beta_q_th

    # Analytic standing-wave envelopes |sin(beta_q z)| and |cos(beta_q z)|
    I1_norm = I1_env / max(float(np.max(I1_env)), 1e-30)
    V1_norm = V1_env / max(float(np.max(V1_env)), 1e-30)
    I1_th_norm = np.abs(np.sin(beta_q_th * z_drift))
    V1_th_norm = np.abs(np.cos(beta_q_th * z_drift))

    # Measure beat half-wavelength L_beat = pi / beta_q from the first current peak and first node
    # Fit beta_q by maximizing correlation with |sin(beta_q z)|
    bq_grid = np.linspace(0.75 * beta_q_th, 1.25 * beta_q_th, 2000)
    sin_mat = np.abs(np.sin(bq_grid[:, None] * z_drift[None, :]))
    sin_mat /= np.linalg.norm(sin_mat, axis=1, keepdims=True)
    i_unit = I1_norm / max(float(np.linalg.norm(I1_norm)), 1e-30)
    k_best = int(np.argmax(sin_mat @ i_unit))
    beta_q_fit = float(bq_grid[k_best])
    L_beat_fit = float(np.pi / beta_q_fit)
    R_beat_fit = float(beta_q_fit * float(u0) * (gamma0**1.5) / wp)

    # Snapshot of 2D fields (rho1, J1x, Ex) at end of run
    rho1_2d = beam.charge_density_field()[:, :, 0].astype(np.float64)
    J1_2d = beam.current_density_field()[:, :, 0].astype(np.float64)
    Ex_2d = solver.E[: solver.Nx, : solver.Ny, 0, "x"].astype(np.float64)

    min_feature = float(min(d_gap, b_half))
    cells_across = min(d_gap / dx, float(b_half) / dy)

    out = {
        "z_drift": z_drift.astype(np.float64),
        "I1_norm": I1_norm.astype(np.float64),
        "V1_norm": V1_norm.astype(np.float64),
        "I1_th_norm": I1_th_norm.astype(np.float64),
        "V1_th_norm": V1_th_norm.astype(np.float64),
        "beta_q_th": float(beta_q_th),
        "beta_q_fit": float(beta_q_fit),
        "L_beat_th": float(L_beat_th),
        "L_beat_fit": float(L_beat_fit),
        "R_th": float(R_th),
        "R_beat_fit": float(R_beat_fit),
        "M_gap": float(M_gap),
        "theta_g": float(theta_g),
        "V0": float(V0),
        "Vm": float(V_m),
        "x_coords": (grid.x[: solver.Nx] * 1e3).astype(np.float64),
        "y_coords": (grid.y[: solver.Ny] * 1e3).astype(np.float64),
        "s_coords_mm": (beam.s_coords * 1e3).astype(np.float64),
        "w_axial": beam.w_axial.astype(np.float64),
        "rho1_2d": rho1_2d,
        "J1_2d": J1_2d,
        "Ex_2d": Ex_2d,
        "dx_background": float(max(dx, dy)),
        "dx_min": float(min(dx, dy)),
        "min_feature": min_feature,
        "cells_across_feature": float(cells_across),
        "cpw": (float(u0) / float(f_mod)) / dx,
        "n_cells": int(nx) * int(ny),
        "n_steps": int(nt_total),
    }
    if record_movie and len(movie_times) > 0:
        out["movie_times"] = np.asarray(movie_times, dtype=np.float64)
        out["movie_rho"] = np.stack(movie_frames_rho, axis=0)
        out["movie_Ex"] = np.stack(movie_frames_Ex, axis=0)
    return out


def run_chu_energy_and_slow_wave_audit(
    b_half=B_BEAM_REF,
    a_half=A_TUNNEL_REF,
    f0=F0_REF,
    u0=U0_REF,
    wp=WP_REF,
    nx=32,
    ny=32,
    nt=1800,
):
    """
    1. Evaluate Chu kinetic voltage ``V_1(z)``, current ``J_{1z}(z)``, and kinetic
       power ``P_k(z)`` for a pure fast space-charge wave (``P_k > 0``) and a pure
       slow space-charge wave (``P_k < 0``).
    2. Evolve the coupled 2D FIT + ``LinearizedBeam`` system in a lossless periodic
       drift guide for ``nt`` steps and record ``U_EM(t)``, ``U_kin(t)``, and
       ``U_tot(t) = U_EM(t) + U_kin(t)``.
    3. Compute the nonlinear Lagrangian (Bessel) harmonic currents ``I_n(X) / I_0 =
       2 J_n(n X)`` vs linear cold-fluid ``I_1(X) / I_0 = X`` and two-particle
       phase-space trajectories past the overtaking limit ``X = 1``.
    """
    omega0 = 2.0 * np.pi * float(f0)
    beta = omega0 / float(u0)
    Lx = 2.0 * np.pi / beta
    dx = Lx / int(nx)
    dy = (2.0 * float(a_half)) / int(ny)

    grid = wakis.GridFIT3D(
        0.0,
        Lx,
        -float(a_half),
        float(a_half),
        -0.5 * dx,
        0.5 * dx,
        int(nx),
        int(ny),
        1,
        verbose=0,
    )
    solver = wakis.SolverFIT3D(
        grid,
        bc_low=["periodic", "pec", "periodic"],
        bc_high=["periodic", "pec", "periodic"],
        cfln=0.45,
        verbose=0,
    )
    beam = LinearizedBeam(
        u0=u0,
        omega_p=wp,
        axis="x",
        half_thickness=b_half,
        solver=solver,
    )

    R_th = reduction_factor_planar(omega0, u0, b_half, a=a_half)
    omega_fast, omega_slow = space_charge_wave_dispersion(
        beta, u0, wp, R=R_th
    )
    omega_q = 0.5 * (omega_fast - omega_slow)

    s = beam.s_coords
    v_amp = 1.0e4
    v_wave = v_amp * np.cos(beta * s)[:, None] * np.ones((beam.M, beam.N_lines))

    # Pure fast wave: rho1 = +(beta * rho0 / omega_q) v1
    beam.v1[:] = v_wave
    beam.rho1[:] = (beta / omega_q) * beam.rho0 * v_wave
    beam.J1[:] = beam.rho0 * beam.v1 + beam.u0 * beam.rho1
    V1_fast = np.mean(beam.kinetic_voltage(), axis=1)
    I1_fast = np.sum(beam.J1 * beam.dA_perp, axis=1)
    Pk_fast_z = beam.kinetic_power_profile()
    Pk_fast_mean = beam.kinetic_power()

    # Pure slow wave: rho1 = -(beta * rho0 / omega_q) v1
    beam.v1[:] = v_wave
    beam.rho1[:] = -(beta / omega_q) * beam.rho0 * v_wave
    beam.J1[:] = beam.rho0 * beam.v1 + beam.u0 * beam.rho1
    V1_slow = np.mean(beam.kinetic_voltage(), axis=1)
    I1_slow = np.sum(beam.J1 * beam.dA_perp, axis=1)
    Pk_slow_z = beam.kinetic_power_profile()
    Pk_slow_mean = beam.kinetic_power()

    # Coupled time-domain Chu-Poynting energy conservation run
    solver.reset_fields()
    y_nodes = grid.y[: solver.Ny]
    Es_prof, _, _, _, _, _ = planar_tm_eigenmode(
        y_nodes, omega0, u0, b_half, a_half
    )
    beam.v1[:] = (
        v_amp
        * np.sin(beta * s)[:, None]
        * Es_prof[beam.keep_lines][None, :]
    )

    itA_arr = grid.itA.toarray()
    tA = np.divide(1.0, itA_arr, out=np.zeros_like(itA_arr), where=itA_arr != 0)
    dV_e = grid.L.toarray() * tA
    iA_arr = grid.iA.toarray()
    da = np.divide(1.0, iA_arr, out=np.zeros_like(iA_arr), where=iA_arr != 0)
    dV_h = grid.tL.toarray() * da
    ieps = solver.ieps.toarray()
    imu = solver.imu.toarray()
    eps_vec = np.divide(1.0, ieps, out=np.zeros_like(ieps), where=ieps != 0)
    mu_vec = np.divide(1.0, imu, out=np.zeros_like(imu), where=imu != 0)

    def _em_energy():
        E_arr = solver.E.toarray()
        H_arr = solver.H.toarray()
        u_e = 0.5 * float(np.sum(eps_vec * np.abs(E_arr) ** 2 * dV_e))
        u_h = 0.5 * float(np.sum(mu_vec * np.abs(H_arr) ** 2 * dV_h))
        return u_e + u_h

    u_em = np.zeros(nt, dtype=np.float64)
    u_kin = np.zeros(nt, dtype=np.float64)
    u_tot = np.zeros(nt, dtype=np.float64)
    t_arr = np.arange(nt, dtype=np.float64) * solver.dt

    for n in range(nt):
        solver.one_step()
        E_prev = solver.E.toarray().copy()
        u_h_half = 0.5 * float(
            np.sum(mu_vec * np.abs(solver.H.toarray()) ** 2 * dV_h)
        )
        u_k_half = beam.kinetic_energy()
        if n > 0:
            u_e_half = 0.25 * float(
                np.sum(
                    eps_vec
                    * (np.abs(E_prev) ** 2 + np.abs(E_old) ** 2)
                    * dV_e
                )
            )
            u_em[n] = u_e_half + u_h_half
            u_kin[n] = u_k_half
            u_tot[n] = u_em[n] + u_kin[n]
        E_old = E_prev

    u_em[0] = u_em[1]
    u_kin[0] = u_kin[1]
    u_tot[0] = u_tot[1]
    rel_drift = float((np.max(u_tot) - np.min(u_tot)) / np.mean(u_tot))

    # Analytical vs PIC/Lagrangian bunching parameter X sweep
    X_arr = np.linspace(0.0, 3.0, 300)
    I1_linear = X_arr.copy()
    I1_bessel = np.abs(2.0 * bessel_jn(1, X_arr))
    I2_bessel = np.abs(2.0 * bessel_jn(2, 2.0 * X_arr))
    I3_bessel = np.abs(2.0 * bessel_jn(3, 3.0 * X_arr))

    return {
        "s_mm": (s * 1e3).astype(np.float64),
        "V1_fast": V1_fast.astype(np.float64),
        "I1_fast": I1_fast.astype(np.float64),
        "Pk_fast_z": Pk_fast_z.astype(np.float64),
        "Pk_fast_mean": float(Pk_fast_mean),
        "V1_slow": V1_slow.astype(np.float64),
        "I1_slow": I1_slow.astype(np.float64),
        "Pk_slow_z": Pk_slow_z.astype(np.float64),
        "Pk_slow_mean": float(Pk_slow_mean),
        "omega_fast": float(omega_fast),
        "omega_slow": float(omega_slow),
        "omega_q": float(omega_q),
        "t_ns": (t_arr * 1e9).astype(np.float64),
        "u_em": u_em,
        "u_kin": u_kin,
        "u_tot": u_tot,
        "rel_energy_drift": rel_drift,
        "X_arr": X_arr.astype(np.float64),
        "I1_linear": I1_linear.astype(np.float64),
        "I1_bessel": I1_bessel.astype(np.float64),
        "I2_bessel": I2_bessel.astype(np.float64),
        "I3_bessel": I3_bessel.astype(np.float64),
        "dx_background": float(max(dx, dy)),
        "dx_min": float(min(dx, dy)),
        "min_feature": float(b_half),
        "cells_across_feature": float(b_half) / dy,
        "cpw": (float(u0) / float(f0)) / dx,
        "n_cells": int(nx) * int(ny),
        "n_steps": int(nt),
    }


# ---------------------------------------------------------------------------
# Schematic setup builders & figure generation suite (00a/00b + 14 figures)
# ---------------------------------------------------------------------------


class _RegionBox:
    """Axis-aligned box region with ``bounds``, ``material``, and ``name`` for ``wakis.schematic``."""

    def __init__(self, low, high, name: str, material="LinearizedBeam"):
        self._low = np.asarray(low, dtype=float)
        self._high = np.asarray(high, dtype=float)
        self.name = str(name)
        self.material = material

    @property
    def bounds(self):
        return self._low, self._high


def _build_bloch_2d_setup(primary: dict):
    """Construct live SolverFIT3D, beam box, source, and probe records for the 2D Bloch setup."""
    nx = int(primary.get("nx", 64))
    ny = int(primary.get("ny", 128))
    b_half = float(primary.get("b_half", B_BEAM_REF))
    a_half = float(primary.get("a_half", A_TUNNEL_REF))
    f0 = float(primary.get("f0", F0_REF))
    u0 = float(primary.get("u0", U0_REF))
    lx = u0 / f0
    dx = lx / nx
    beta = 2.0 * np.pi * f0 / u0

    grid = wakis.GridFIT3D(
        0.0,
        lx,
        -a_half,
        a_half,
        -0.5 * dx,
        0.5 * dx,
        nx,
        ny,
        1,
        verbose=0,
    )
    solver = wakis.SolverFIT3D(
        grid,
        bc_low=["bloch", "pec", "periodic"],
        bc_high=["bloch", "pec", "periodic"],
        k_bloch=[-beta, 0.0, 0.0],
        dtype=np.complex128,
        cfln=0.75,
        verbose=0,
    )
    beam_box = _RegionBox(
        [0.0, -b_half, -0.5 * dx],
        [lx, b_half, 0.5 * dx],
        name="Drifting e⁻ sheet beam (b = 0.5 mm, u₀ = 0.16c, V₀ = 6.70 kV, f_p = 1.35 GHz)",
    )
    src_rec = {
        "kind": "beam",
        "type_name": "LinearizedBeam",
        "short_word": "e⁻ beam",
        "desc": "LinearizedBeam (u₀ = 0.16c, V₀ = 6.70 kV, f_p = 1.35 GHz, λ_e = 9.59 mm)",
        "label": "LinearizedBeam (u₀ = 0.16c, V₀ = 6.70 kV, f_p = 1.35 GHz)",
        "pos_str": "y ∈ [−0.50 mm, +0.50 mm], x ∈ [0 mm, 9.59 mm]",
        "polarization": "J₁ₓ, E_x",
        "h_bounds": (0.0, lx),
        "v_bounds": (0.0, 0.0),
        "k_vec": (1.0, 0.0, 0.0),
        "pulse": {
            "f0": f0,
            "fwidth": 2.70e9,
            "t_peak": 0.40e-9,
            "t_end": 1.60e-9,
            "src_time": None,
        },
    }
    mon_slice = {
        "kind": "plane",
        "type_name": "EigenmodeSlice",
        "short_word": "TM slice",
        "desc": f"TM₁₀ mode slice E_x(y), E_y(y), J₁ₓ(y) (x = {0.25 * lx * 1e3:.2f} mm)",
        "label": f"TM₁₀ eigenmode slice (x = {0.25 * lx * 1e3:.2f} mm)",
        "pos_str": f"x = {0.25 * lx * 1e3:.2f} mm, y ∈ [−2.0 mm, +2.0 mm]",
        "freq_str": "6 freqs in [1.5, 10.0] GHz",
        "frequencies": np.array([1.5e9, 3.0e9, 5.0e9, 7.0e9, 8.5e9, 10.0e9]),
        "h_bounds": (0.25 * lx, 0.25 * lx),
        "v_bounds": (-a_half, a_half),
    }
    probes = [
        {
            "pos": (0.25 * lx, 0.0, 0.0),
            "label": f"Doppler doublet FFT probe E_x(x_p, 0, t) (x_p = {0.25 * lx * 1e3:.2f} mm)",
        }
    ]
    return solver, [beam_box], [src_rec], [mon_slice], probes


def _build_drift_tube_setup(bunch: dict):
    """Construct live SolverFIT3D, structures, sources, and monitors for the 2D klystron drift tube."""
    x_mm = np.asarray(bunch["x_coords"], dtype=float)
    y_mm = np.asarray(bunch["y_coords"], dtype=float)
    nx = len(x_mm)
    ny = len(y_mm)
    l_tube = 0.115
    a_half = A_TUNNEL_REF
    b_half = 0.8e-3
    dx = l_tube / nx
    n_pml = 6
    ramp_cells = 10

    grid = wakis.GridFIT3D(
        0.0,
        l_tube,
        -a_half,
        a_half,
        -0.5 * dx,
        0.5 * dx,
        nx,
        ny,
        1,
        verbose=0,
    )
    solver = wakis.SolverFIT3D(
        grid,
        bc_low=["cpml", "pec", "periodic"],
        bc_high=["cpml", "pec", "periodic"],
        n_pml=n_pml,
        cfln=0.5,
        verbose=0,
    )
    x_taper_start = l_tube - (n_pml + ramp_cells) * dx
    x_pml_high = l_tube - n_pml * dx
    x_pml_low = n_pml * dx
    s_in = (n_pml + 3) * dx
    d_gap = 4.0 * dx
    s_gap = s_in + 0.5 * d_gap

    beam_box = _RegionBox(
        [x_pml_low, -b_half, -0.5 * dx],
        [x_taper_start, b_half, 0.5 * dx],
        name="Drifting e⁻ beam (|y| ≤ 0.8 mm, u₀ = 0.16c)",
    )
    taper_box = _RegionBox(
        [x_taper_start, -b_half, -0.5 * dx],
        [x_pml_high, b_half, 0.5 * dx],
        name="ρ₀ → 0",
    )
    gap_src = {
        "kind": "sheet",
        "type_name": "RFModulatingGap",
        "short_word": "RF gap",
        "desc": f"RF modulating gap V_m = 100 V, f_mod = 5.0 GHz (z_g = {s_gap * 1e3:.2f} mm, d_g = {d_gap * 1e3:.2f} mm)",
        "label": f"RF modulating gap (z_g = {s_gap * 1e3:.2f} mm)",
        "pos_str": f"z = {s_gap * 1e3:.2f} mm",
        "polarization": "v₁(t) = Δv sin(ωt)",
        "h_bounds": (s_gap, s_gap),
        "v_bounds": (-b_half, b_half),
        "k_vec": (1.0, 0.0, 0.0),
        "pulse": None,
    }
    l_beat_th = float(bunch["L_beat_th"])
    z_q4 = s_gap + 0.5 * l_beat_th
    z_q2 = s_gap + l_beat_th
    monitors = [
        {
            "kind": "plane",
            "type_name": "BunchingPeakMonitor",
            "short_word": "λ_q/4 peak",
            "desc": f"Klystron bunching peak λ_q/4 (z = {z_q4 * 1e3:.1f} mm)",
            "label": f"λ_q/4 current peak (z = {z_q4 * 1e3:.1f} mm)",
            "pos_str": f"z = {z_q4 * 1e3:.1f} mm",
            "freq_str": "f = 5.0 GHz",
            "frequencies": np.array([F0_REF]),
            "h_bounds": (z_q4, z_q4),
            "v_bounds": (-b_half, b_half),
        },
        {
            "kind": "plane",
            "type_name": "CurrentNodeMonitor",
            "short_word": "λ_q/2 node",
            "desc": f"First current node λ_q/2 (z = {z_q2 * 1e3:.1f} mm)",
            "label": f"λ_q/2 current node (z = {z_q2 * 1e3:.1f} mm)",
            "pos_str": f"z = {z_q2 * 1e3:.1f} mm",
            "freq_str": "f = 5.0 GHz",
            "frequencies": np.array([F0_REF]),
            "h_bounds": (z_q2, z_q2),
            "v_bounds": (-b_half, b_half),
        },
    ]
    return solver, [beam_box, taper_box], [gap_src], monitors


def _build_tunnel_3d_setup(beam3d: dict):
    """Construct live SolverFIT3D and 3D beam box for the 3D square PEC tunnel."""
    x_n = np.asarray(beam3d["x_nodes"], dtype=float)
    y_n = np.asarray(beam3d["y_nodes"], dtype=float)
    nx = len(x_n)
    ny = len(y_n)
    nz = max(16, int(round(float(beam3d["n_cells"]) / max(1, nx * ny))))
    b_half = float(beam3d["b_half"])
    a_half = float(beam3d["a_half"])
    f0 = float(beam3d.get("f0", F0_REF))
    lz = U0_REF / f0
    beta = 2.0 * np.pi * f0 / U0_REF

    grid = wakis.GridFIT3D(
        -a_half,
        a_half,
        -a_half,
        a_half,
        0.0,
        lz,
        nx,
        ny,
        nz,
        verbose=0,
    )
    solver = wakis.SolverFIT3D(
        grid,
        bc_low=["pec", "pec", "bloch"],
        bc_high=["pec", "pec", "bloch"],
        k_bloch=[0.0, 0.0, -beta],
        dtype=np.complex128,
        cfln=0.78,
        verbose=0,
    )
    beam_box = _RegionBox(
        [-b_half, -b_half, 0.0],
        [b_half, b_half, lz],
        name=f"3D e⁻ beam (b = {b_half * 1e3:.1f} mm)",
    )
    probe = [{"pos": (0.0, 0.0, 0.25 * lz), "label": "Axis probe E_z(0, 0)"}]
    return solver, [beam_box], probe


def _add_pec_wall_tags_2d(ax, label_top: str = "+y: PEC wall", label_bot: str = "−y: PEC wall"):
    """Add high-contrast top/bottom PEC wall tags with white stroke halos on a 2D field map."""
    halo = schematic._text_halo(2.5)
    ax.text(
        0.50,
        0.96,
        label_top,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8.8,
        fontweight="bold",
        color=figstyle.INK,
        path_effects=halo,
        zorder=9,
    )
    ax.text(
        0.50,
        0.04,
        label_bot,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=8.8,
        fontweight="bold",
        color=figstyle.INK,
        path_effects=halo,
        zorder=9,
    )


def render_all_figures(tasks_data, fig_dir):
    """
    Render setup schematics (``00a``, ``00b``, ``00_setup_table.md``) and all 14
    diagnostic and validation figures for Tutorial 8.1 into ``fig_dir``.
    """
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    primary = tasks_data["disp_res_fine"]
    kw_task = tasks_data["kw_and_freq_sweep"]
    thick_task = tasks_data["thickness_sweep"]
    beam3d = tasks_data["beam_3d_tunnel"]
    bunch = tasks_data["drift_tube_bunching"]
    chu = tasks_data["chu_energy_and_slow_wave"]

    s_bloch, structs_bloch, srcs_bloch, mons_bloch, probes_bloch = _build_bloch_2d_setup(primary)
    s_bunch, structs_bunch, srcs_bunch, mons_bunch = _build_drift_tube_setup(bunch)
    s_3d, structs_3d, probes_3d = _build_tunnel_3d_setup(beam3d)

    with figstyle.apply_style():
        # ------------------------------------------------------------------
        # Fig 00a & 00b: Setup schematic (Figure A) + Mesh & excitation (Figure B) + table
        # ------------------------------------------------------------------
        schematic.save_setup_figures(
            s_bloch,
            fig_dir,
            plane="xy",
            unit="mm",
            structures=structs_bloch,
            sources=srcs_bloch,
            monitors=mons_bloch,
            probes=probes_bloch,
            wavelength=LAMBDA_E_REF,
            freq=F0_REF,
            min_feature=float(primary.get("b_half", B_BEAM_REF)),
            title_geometry="Tutorial 8.1 — Setup: sheet electron beam, PEC drift tube, Bloch cell & probes",
            title_mesh="Tutorial 8.1 — Discretization and space-charge wave excitation",
            variations=[
                {
                    "name": "Tasks 0–2: 2D Bloch dispersion & convergence (§5, §7–8)",
                    "details": (
                        "2D Bloch cell L_x = λ_e = 9.59 mm, 2a = 4.0 mm, PEC ±y walls, "
                        "sheet e⁻ beam b = 0.5 mm (u₀ = 0.16c, V₀ = 6.70 kV, f_p = 1.35 GHz); "
                        "grids 16×32, 24×48, 64×128"
                    ),
                },
                {
                    "name": "Tasks 3–4: Frequency & filling-ratio sweeps (§7)",
                    "details": (
                        "2D Bloch cell (24×40); f₀ ∈ {1.5, 3.0, 5.0, 7.0, 8.5, 10.0} GHz "
                        "and b/a ∈ {0.25, 0.375, 0.50, 0.75, 1.00}"
                    ),
                },
                {
                    "name": "Task 5: 3D square PEC drift tunnel (§7)",
                    "details": (
                        "3D domain 4.0 mm × 4.0 mm × 9.59 mm (40×40×24), PEC ±x, ±y walls, "
                        "Bloch ±z, pencil beam b = 0.5 mm (f_p = 1.65 GHz)"
                    ),
                },
                {
                    "name": "Task 6: 2D klystron velocity-modulated drift tube (§7)",
                    "details": (
                        "Open tube L_z = 115 mm × 4.0 mm (360×32), CPML ±z (N_pml = 6, 1.92 mm) "
                        "+ 10-cell ρ₀→0 taper, PEC ±y walls, RF gap V_m = 100 V at z_g = 3.51 mm, beam b = 0.8 mm"
                    ),
                },
                {
                    "name": "Task 7: Chu kinetic power & energy audit (§3, §7)",
                    "details": (
                        "Periodic ±x, PEC ±y (48×48, 2400 steps); fast wave (P_k > 0) vs "
                        "slow wave (P_k < 0) and coupled Chu–Poynting conservation"
                    ),
                },
            ],
        )

        # ------------------------------------------------------------------
        # Fig 01: Drift-tube & velocity-modulating gap schematic + 2D/3D cross-section
        # ------------------------------------------------------------------
        fig, (ax0, ax1) = figstyle.new_figure(
            width_in=figstyle.FULL_WIDTH_IN,
            height_in=3.4,
            ncols=2,
            nrows=1,
            gridspec_kw={"width_ratios": [1.65, 1.0]},
            constrained_layout=False,
        )
        fig.subplots_adjust(left=0.08, right=0.98, bottom=0.18, top=0.86, wspace=0.28)

        l_mm = 115.0
        a_mm = A_TUNNEL_REF * 1e3
        b_mm = 0.8
        ax0.add_patch(
            mpatches.Rectangle(
                (0.0, -a_mm), l_mm, 2.0 * a_mm, facecolor="#F8FAFC", edgecolor="none", zorder=1
            )
        )
        ax0.add_patch(
            mpatches.Rectangle(
                (1.92, -b_mm),
                l_mm - 5.12 - 1.92,
                2.0 * b_mm,
                facecolor="#DCE8F2",
                edgecolor="#0072B2",
                lw=1.3,
                alpha=0.88,
                zorder=2,
                label=r"e⁻ beam ($|y|\leq 0.8\;\mathrm{mm},\;u_0=0.16c$)",
            )
        )
        ax0.add_patch(
            mpatches.Rectangle(
                (l_mm - 5.12, -b_mm),
                3.20,
                2.0 * b_mm,
                facecolor="#F5E4C8",
                edgecolor="#E69F00",
                lw=1.1,
                alpha=0.85,
                zorder=2,
                label=r"$\rho_0(z)\to 0$ taper (10 cells)",
            )
        )
        schematic.overlay_geometry(
            ax0,
            s_bunch,
            plane="xy",
            unit="mm",
            structures=structs_bunch,
            sources=srcs_bunch,
            monitors=mons_bunch,
            show_boundaries=True,
            show_materials=False,
            show_sources=True,
            show_monitors=True,
        )
        ax0.axhline(+a_mm, color=figstyle.INK, lw=2.8)
        ax0.axhline(-a_mm, color=figstyle.INK, lw=2.8)
        _add_pec_wall_tags_2d(ax0, "+y: PEC drift tube", "−y: PEC drift tube")
        ax0.annotate(
            r"$\mathbf{u}_0 = 0.16c$ ($6.7\;\mathrm{kV}$)",
            xy=(36.0, 0.0),
            xytext=(11.0, 0.0),
            arrowprops={"arrowstyle": "-|>", "lw": 1.6, "color": "#0072B2"},
            fontsize=8.8,
            va="center",
            color="#0072B2",
            fontweight="bold",
            path_effects=schematic._text_halo(2.5),
            zorder=8,
        )
        ax0.text(
            24.8,
            1.22,
            r"$\lambda_q/4$ peak",
            ha="center",
            va="center",
            fontsize=8.6,
            fontweight="bold",
            color="#009E73",
            path_effects=schematic._text_halo(2.5),
            zorder=8,
        )
        ax0.text(
            46.0,
            -1.22,
            r"$\lambda_q/2$ node",
            ha="center",
            va="center",
            fontsize=8.6,
            fontweight="bold",
            color="#009E73",
            path_effects=schematic._text_halo(2.5),
            zorder=8,
        )
        ax0.set(
            xlim=(0.0, l_mm),
            ylim=(-a_mm * 1.15, a_mm * 1.15),
            xlabel="Axial drift coordinate z [mm]",
            ylabel="Transverse y [mm]",
            title="(a) Klystron drift tube (Task 6)",
        )
        figstyle.place_legend_below(ax0, ncol=1, y_offset=-0.22)

        # Panel (b): Transverse cross-section (2D sheet vs 3D square vs 3D circular)
        ax1.add_patch(
            mpatches.Rectangle(
                (-a_mm, -a_mm),
                2.0 * a_mm,
                2.0 * a_mm,
                facecolor="#F8FAFC",
                edgecolor=figstyle.INK,
                lw=2.6,
                label="3D square PEC tunnel",
            )
        )
        schematic.draw_mesh(
            ax1,
            s_3d,
            plane="xy",
            unit="mm",
            every=4,
            show_cpw=False,
        )
        ax1.add_patch(
            mpatches.Circle(
                (0.0, 0.0),
                a_mm,
                fill=False,
                edgecolor="#475569",
                ls="--",
                lw=1.5,
                label="Circular tube ($R_t = 2$ mm)",
            )
        )
        ax1.add_patch(
            mpatches.Rectangle(
                (-a_mm, -0.5),
                2.0 * a_mm,
                1.0,
                facecolor="#D5ECE4",
                alpha=0.55,
                edgecolor="#009E73",
                ls=":",
                lw=1.5,
                label="2D sheet beam ($|y|\\leq 0.5$ mm)",
            )
        )
        ax1.add_patch(
            mpatches.Circle(
                (0.0, 0.0),
                0.5,
                facecolor="#DCE8F2",
                alpha=0.85,
                edgecolor="#0072B2",
                lw=1.6,
                label="3D beam ($R_b = 0.5$ mm)",
            )
        )
        ax1.set(
            xlim=(-a_mm * 1.15, a_mm * 1.15),
            ylim=(-a_mm * 1.15, a_mm * 1.15),
            aspect="equal",
            xlabel="Transverse x [mm]",
            ylabel="Transverse y [mm]",
            title="(b) 2D / 3D cross-section",
        )
        figstyle.place_legend_below(ax1, ncol=1, y_offset=-0.22)
        figstyle.save(fig, fig_dir / "01_drift_tube_geometry.png", close=True, auto_legend_below=False)

        # ------------------------------------------------------------------
        # Fig 02: Mesh zoom & transverse/axial profile weights (3 panels)
        # ------------------------------------------------------------------
        fig, (ax_m, ax0, ax1) = figstyle.new_figure(
            width_in=figstyle.FULL_WIDTH_IN,
            height_in=3.3,
            ncols=3,
            nrows=1,
            constrained_layout=False,
        )
        fig.subplots_adjust(left=0.07, right=0.98, bottom=0.19, top=0.86, wspace=0.38)

        ax_m.add_patch(
            mpatches.Rectangle(
                (0.0, -2.0), 9.593, 4.0, facecolor="#F8FAFC", edgecolor=figstyle.INK, lw=1.4
            )
        )
        ax_m.add_patch(
            mpatches.Rectangle(
                (0.0, -0.5),
                9.593,
                1.0,
                facecolor="#DCE8F2",
                edgecolor="#0072B2",
                lw=1.2,
                alpha=0.8,
                label="Beam (|y| ≤ 0.5 mm)",
            )
        )
        schematic.draw_mesh(
            ax_m,
            s_bloch,
            plane="xy",
            unit="mm",
            every=4,
            wavelength=LAMBDA_E_REF,
            zoom_inset=False,
            show_cpw=False,
        )
        ax_m.set(
            xlim=(0.0, 9.593),
            ylim=(-2.0, 2.0),
            xlabel="Axial x [mm]",
            ylabel="Transverse y [mm]",
            title="(a) Bloch cell mesh",
        )

        for idx_l, lvl in enumerate(("coarse", "medium", "fine")):
            d = tasks_data[f"disp_res_{lvl}"]
            y_mm = d["y_nodes"] * 1e3
            w_y = d["w_perp_y"]
            st_l = figstyle.series_style(idx_l, ms=4.2)
            ax0.plot(
                y_mm,
                w_y,
                label=f"{lvl.capitalize()} (N_b={d['cells_across_feature']:.0f})",
                **st_l,
            )
        y_ref = np.linspace(-2.0, 2.0, 500)
        ax0.plot(
            y_ref,
            (np.abs(y_ref) <= 0.5).astype(float),
            color=figstyle.INK,
            lw=1.2,
            ls=":",
            marker="none",
            label="Exact |y| ≤ 0.5 mm",
        )
        ax0.set(
            xlim=(-1.2, 1.2),
            ylim=(-0.05, 1.12),
            xlabel="Transverse y [mm]",
            ylabel=r"Overlap weight $w_\perp(y)$",
            title=r"(b) Beam weight $w_\perp(y)$",
        )
        figstyle.spectrum_grid(ax0)
        figstyle.place_legend_below(ax0, ncol=1, y_offset=-0.24)

        s_mm = bunch["s_coords_mm"]
        w_ax = bunch["w_axial"]
        ax1.plot(
            s_mm,
            w_ax,
            label=r"Axial weight $w_s(z)$",
            **figstyle.series_style(0, markevery=6),
        )
        ax1.axvspan(
            s_mm[-1],
            115.0,
            facecolor="#CBD5E1",
            edgecolor="#475569",
            hatch="///",
            alpha=0.45,
            label="Outflow CPML (ρ₀ = 0)",
        )
        ax1.axvspan(
            s_mm[-10],
            s_mm[-1],
            color="#F5E4C8",
            alpha=0.65,
            label="10-cell ρ₀ → 0 taper",
        )
        ax1.set(
            xlim=(float(s_mm[-35]), 115.0),
            ylim=(-0.05, 1.12),
            xlabel="Axial z near collector [mm]",
            ylabel=r"Normalized $\rho_0(z)/\rho_{0,\mathrm{peak}}$",
            title="(c) Collector density ramp",
        )
        figstyle.spectrum_grid(ax1)
        figstyle.place_legend_below(ax1, ncol=1, y_offset=-0.26)
        figstyle.save(fig, fig_dir / "02_mesh_and_profile_zoom.png", close=True, auto_legend_below=False)

        # ------------------------------------------------------------------
        # Fig 03: Transverse TM space-charge wave eigenmode profiles
        # ------------------------------------------------------------------
        fig, (ax0, ax1) = figstyle.new_figure(
            width_in=figstyle.FULL_WIDTH_IN,
            height_in=3.3,
            ncols=2,
            nrows=1,
            constrained_layout=False,
        )
        fig.subplots_adjust(left=0.09, right=0.98, bottom=0.18, top=0.88, wspace=0.32)

        y_n_mm = primary["y_nodes"] * 1e3
        y_e_mm = primary["y_edges"] * 1e3
        y_dense = np.linspace(-A_TUNNEL_REF, A_TUNNEL_REF, 400)
        Es_d, Ey_d, J1_d, _, _, _ = planar_tm_eigenmode(
            y_dense, OMEGA0_REF, U0_REF, B_BEAM_REF, A_TUNNEL_REF
        )

        ax0.plot(
            y_dense * 1e3,
            Es_d,
            label=r"Analytic $E_z(y)$",
            **figstyle.series_style(0, with_marker=False),
        )
        ax0.plot(
            y_n_mm[::2],
            primary["Ex_fit"][::2],
            label="Wakis 2D FIT $E_z(y)$",
            **figstyle.series_style(1, with_line=False, ms=4.5),
        )
        ax0.plot(
            y_dense * 1e3,
            J1_d,
            label=r"Analytic $J_{1z}(y)/J_{\max}$",
            **figstyle.series_style(2, with_marker=False),
        )
        ax0.plot(
            y_n_mm[::2],
            primary["J1_fit"][::2],
            label="Wakis 2D FIT $J_{1z}(y)$",
            **figstyle.series_style(2, with_line=False, ms=4.2),
        )
        ax0.axvspan(-0.5, 0.5, color="#DCE8F2", alpha=0.55, label="Beam (|y| ≤ 0.5 mm)")
        ax0.set(
            xlabel="Transverse y [mm]",
            ylabel="Normalized amplitude",
            title=r"(a) Axial $E_z(y)$ & current $J_{1z}(y)$",
        )
        figstyle.spectrum_grid(ax0)
        figstyle.place_legend_below(ax0, ncol=1, y_offset=-0.22)

        ax1.plot(
            y_dense * 1e3,
            Ey_d,
            label=r"Analytic $E_y(y)/E_z(0)$",
            **figstyle.series_style(4, with_marker=False),
        )
        ax1.plot(
            y_e_mm[::2],
            primary["Ey_fit"][::2],
            label="Wakis 2D FIT $E_y(y)/E_z(0)$",
            **figstyle.series_style(1, with_line=False, ms=4.5),
        )
        ax1.axvspan(-0.5, 0.5, color="#DCE8F2", alpha=0.55, label="Beam (|y| ≤ 0.5 mm)")
        ax1.axhline(0.0, color=figstyle.INK, ls=":", lw=1.0, marker="none")
        ax1.set(
            xlabel="Transverse y [mm]",
            ylabel=r"Transverse field $E_y(y)/E_z(0)$",
            title=r"(b) Transverse fringing $E_y(y)$",
        )
        figstyle.spectrum_grid(ax1)
        figstyle.place_legend_below(ax1, ncol=1, y_offset=-0.22)
        figstyle.save(fig, fig_dir / "03_transverse_tm_eigenmodes.png", close=True, auto_legend_below=False)

        # ------------------------------------------------------------------
        # Fig 04: Space-charge wave k-omega dispersion diagram
        # ------------------------------------------------------------------
        fig, (ax0, ax1) = figstyle.new_figure(
            width_in=figstyle.FULL_WIDTH_IN,
            height_in=3.4,
            ncols=2,
            nrows=1,
            constrained_layout=False,
        )
        fig.subplots_adjust(left=0.09, right=0.98, bottom=0.18, top=0.88, wspace=0.34)

        beta_dense = np.linspace(50.0, 1450.0, 300)
        f_beam_dense = (beta_dense * U0_REF) / (2.0 * np.pi) * 1e-9
        R_dense = reduction_factor_planar(
            beta_dense * U0_REF, U0_REF, B_BEAM_REF, A_TUNNEL_REF
        )
        w_fast_d, w_slow_d = space_charge_wave_dispersion(
            beta_dense, U0_REF, WP_REF, R=R_dense
        )
        f_fast_d = w_fast_d / (2.0 * np.pi) * 1e-9
        f_slow_d = w_slow_d / (2.0 * np.pi) * 1e-9
        f_light_d = (beta_dense * c_light) / (2.0 * np.pi) * 1e-9

        betas_fit = kw_task["betas"]
        f_fast_fit = kw_task["omega_fast"] / (2.0 * np.pi) * 1e-9
        f_slow_fit = kw_task["omega_slow"] / (2.0 * np.pi) * 1e-9

        ax0.plot(
            beta_dense,
            f_beam_dense,
            label=r"Beam line $\omega = \beta u_0$",
            **figstyle.series_style(7, with_marker=False),
        )
        ax0.plot(
            beta_dense,
            f_fast_d,
            label=r"Fast $\omega_+ = \beta u_0 + \omega_q$",
            **figstyle.series_style(2, with_marker=False),
        )
        ax0.plot(
            beta_dense,
            f_slow_d,
            label=r"Slow $\omega_- = \beta u_0 - \omega_q$",
            **figstyle.series_style(1, with_marker=False),
        )
        ax0.plot(
            betas_fit,
            f_fast_fit,
            label=r"FIT $\omega_+(\beta)$",
            **figstyle.series_style(2, with_line=False, ms=5.8),
        )
        ax0.plot(
            betas_fit,
            f_slow_fit,
            label=r"FIT $\omega_-(\beta)$",
            **figstyle.series_style(1, with_line=False, ms=5.8),
        )
        ax0.plot(
            beta_dense,
            f_light_d,
            label=r"Light line $\omega = c\beta$",
            **figstyle.series_style(4, with_marker=False),
        )
        ax0.set(
            xlim=(50.0, 1400.0),
            ylim=(0.0, 11.5),
            xlabel=r"Axial wavenumber $\beta$ [rad/m]",
            ylabel="Frequency f [GHz]",
            title=r"(a) $(\beta, \omega)$ dispersion diagram",
        )
        figstyle.spectrum_grid(ax0)
        ax0.legend(loc="upper left", frameon=False, fontsize=8.5)

        df_th = (w_fast_d - beta_dense * U0_REF) / (2.0 * np.pi) * 1e-6  # MHz
        df_fast_fit = (kw_task["omega_fast"] - betas_fit * U0_REF) / (2.0 * np.pi) * 1e-6
        df_slow_fit = (kw_task["omega_slow"] - betas_fit * U0_REF) / (2.0 * np.pi) * 1e-6
        fp_rel_mhz = (WP_REF / (GAMMA0_REF**1.5)) / (2.0 * np.pi) * 1e-6

        ax1.plot(
            beta_dense,
            +df_th,
            label=r"$+\omega_q(\beta)/2\pi$ (Fast)",
            **figstyle.series_style(2, with_marker=False),
        )
        ax1.plot(
            beta_dense,
            -df_th,
            label=r"$-\omega_q(\beta)/2\pi$ (Slow)",
            **figstyle.series_style(1, with_marker=False),
        )
        ax1.plot(
            betas_fit,
            df_fast_fit,
            label="FIT fast shift",
            **figstyle.series_style(2, with_line=False, ms=5.8),
        )
        ax1.plot(
            betas_fit,
            df_slow_fit,
            label="FIT slow shift",
            **figstyle.series_style(1, with_line=False, ms=5.8),
        )
        ax1.axhline(0.0, color=figstyle.INK, ls="--", lw=1.0, marker="none")
        ax1.axhline(
            +fp_rel_mhz,
            color="#475569",
            ls=":",
            lw=1.3,
            marker="none",
            label=r"1D limit $\pm \omega_p / (2\pi\gamma_0^{3/2})$",
        )
        ax1.axhline(-fp_rel_mhz, color="#475569", ls=":", lw=1.3, marker="none")
        ax1.set(
            xlim=(50.0, 1400.0),
            xlabel=r"Axial wavenumber $\beta$ [rad/m]",
            ylabel=r"Doppler shift $(\omega_\pm - \beta u_0)/2\pi$ [MHz]",
            title=r"(b) Co-moving shift $\pm \omega_q(\beta)$",
        )
        figstyle.spectrum_grid(ax1)
        ax1.legend(loc="center right", frameon=False, fontsize=8.5)
        figstyle.save(fig, fig_dir / "04_kw_dispersion_diagram.png", close=True, auto_legend_below=False)

        # ------------------------------------------------------------------
        # Fig 05: Time-domain beating trace & Doppler doublet FFT spectrum
        # ------------------------------------------------------------------
        fig, (ax0, ax1) = figstyle.new_figure(
            width_in=figstyle.FULL_WIDTH_IN,
            height_in=3.3,
            ncols=2,
            nrows=1,
            constrained_layout=False,
        )
        fig.subplots_adjust(left=0.09, right=0.98, bottom=0.18, top=0.88, wspace=0.34)

        t_ns = primary["trace_t"] * 1e9
        ex_re = np.real(primary["trace_Ex"])
        ex_env = np.abs(primary["trace_Ex"])
        scale_e = max(float(np.max(ex_env)), 1e-30)
        ax0.plot(
            t_ns,
            ex_re / scale_e,
            color=figstyle.PALETTE[0],
            lw=0.9,
            marker="none",
            alpha=0.85,
            label=r"$\mathrm{Re}[E_z(t)] / E_{\max}$",
        )
        ax0.plot(
            t_ns,
            ex_env / scale_e,
            color=figstyle.PALETTE[1],
            lw=2.0,
            marker="none",
            label=r"Envelope $|\sin(\omega_q t)|$",
        )
        ax0.plot(t_ns, -ex_env / scale_e, color=figstyle.PALETTE[1], lw=2.0, marker="none")
        ax0.set(
            xlabel="Time t [ns]",
            ylabel="Normalized axial field",
            title=r"(a) Temporal beat ($f_0 = 5.0\;\mathrm{GHz}$)",
        )
        figstyle.spectrum_grid(ax0)
        ax0.legend(loc="upper right", frameon=False, fontsize=8.5)

        f_ghz = primary["fft_freqs"] * 1e-9
        spec_db = 20.0 * np.log10(np.maximum(primary["fft_spec"], 1e-6))
        f_slow_th_ghz = primary["omega_slow_th"] / (2.0 * np.pi) * 1e-9
        f_fast_th_ghz = primary["omega_fast_th"] / (2.0 * np.pi) * 1e-9

        ax1.plot(
            f_ghz,
            spec_db,
            label="Wakis Bloch FIT spectrum",
            **figstyle.series_style(0, with_marker=False),
        )
        ax1.axvline(
            F0_REF * 1e-9,
            color=figstyle.INK,
            ls="--",
            lw=1.3,
            marker="none",
            label=f"Carrier f₀ = {F0_REF * 1e-9:.2f} GHz",
        )
        ax1.axvline(
            f_slow_th_ghz,
            color=figstyle.PALETTE[1],
            ls=":",
            lw=1.7,
            marker="none",
            label=f"Slow f₋ = {f_slow_th_ghz:.3f} GHz",
        )
        ax1.axvline(
            f_fast_th_ghz,
            color=figstyle.PALETTE[2],
            ls=":",
            lw=1.7,
            marker="none",
            label=f"Fast f₊ = {f_fast_th_ghz:.3f} GHz",
        )
        ax1.set(
            xlim=(3.8, 6.2),
            ylim=(-55.0, 4.0),
            xlabel="Frequency f [GHz]",
            ylabel="Spectral magnitude [dB]",
            title=r"(b) Doppler doublet $\omega_\pm = \beta u_0 \pm \omega_q$",
        )
        figstyle.spectrum_grid(ax1)
        ax1.legend(loc="lower center", frameon=False, fontsize=8.5)
        figstyle.save(fig, fig_dir / "05_doppler_splitting_spectra.png", close=True, auto_legend_below=False)

        # ------------------------------------------------------------------
        # Fig 06: Reduction factor R vs beam thickness / filling ratio b/a
        # ------------------------------------------------------------------
        fig, (ax0, ax1) = figstyle.new_figure(
            width_in=figstyle.FULL_WIDTH_IN,
            height_in=3.4,
            ncols=2,
            nrows=1,
            constrained_layout=False,
        )
        fig.subplots_adjust(left=0.09, right=0.98, bottom=0.18, top=0.88, wspace=0.34)

        fill_dense = np.linspace(0.05, 1.0, 200)
        b_dense = fill_dense * A_TUNNEL_REF
        R_2d_curve = np.array(
            [
                reduction_factor_planar(OMEGA0_REF, U0_REF, b_val, A_TUNNEL_REF)
                for b_val in b_dense
            ]
        )
        R_3d_rect_curve = np.array(
            [
                reduction_factor_rectangular(
                    OMEGA0_REF, U0_REF, b_val, b_val, A_TUNNEL_REF, A_TUNNEL_REF
                )
                for b_val in b_dense
            ]
        )
        R_3d_cyl_curve = np.array(
            [
                reduction_factor_cylindrical(OMEGA0_REF, U0_REF, b_val, A_TUNNEL_REF)
                for b_val in b_dense
            ]
        )

        ax0.plot(
            fill_dense,
            R_2d_curve,
            label="2D sheet (Branch–Mihran)",
            **figstyle.series_style(2, with_marker=False),
        )
        ax0.plot(
            thick_task["fill_ratios"],
            thick_task["R_fit"],
            label="Wakis 2D FIT",
            **figstyle.series_style(2, with_line=False, ms=6.0),
        )
        ax0.plot(
            fill_dense,
            R_3d_rect_curve,
            label="3D square tunnel (Theory)",
            **figstyle.series_style(0, with_marker=False),
        )
        ax0.plot(
            [beam3d["b_half"] / beam3d["a_half"]],
            [beam3d["R_3d_fit"]],
            label="Wakis 3D FIT",
            **figstyle.series_style(0, with_line=False, ms=6.5),
        )
        ax0.plot(
            fill_dense,
            R_3d_cyl_curve,
            label="3D circular (Mealy 2022)",
            **figstyle.series_style(1, with_marker=False),
        )
        ax0.set(
            xlim=(0.0, 1.05),
            ylim=(0.0, 0.85),
            xlabel=r"Filling ratio $b / a$ ($a = 2.0\;\mathrm{mm}$)",
            ylabel=r"Reduction factor $R$",
            title=r"(a) $R$ vs filling ratio $b / a$",
        )
        figstyle.spectrum_grid(ax0)
        ax0.legend(loc="upper left", frameon=False, fontsize=8.5)

        tau_b_grid = np.linspace(0.05, 4.0, 200)
        u0_bm = U0_REF
        b_bm = 1.0e-3
        _, g_bm = relativistic_gamma(u0=u0_bm)
        omega_bm = tau_b_grid * g_bm * u0_bm / b_bm
        for idx_r, ratio_ab in enumerate((1.0, 1.5, 2.0, 4.0, np.inf)):
            R_bm = reduction_factor_planar(omega_bm, u0_bm, b_bm, a=ratio_ab * b_bm)
            lbl = r"$a/b = \infty$" if np.isinf(ratio_ab) else f"$a/b = {ratio_ab:g}$"
            ax1.plot(
                tau_b_grid,
                R_bm,
                label=lbl,
                **figstyle.series_style(idx_r, with_marker=False),
            )

        tau_b_fit = (kw_task["betas"] / GAMMA0_REF) * B_BEAM_REF
        ax1.plot(
            tau_b_fit,
            kw_task["R_fit"],
            label="Wakis 2D FIT (a/b = 4)",
            **figstyle.series_style(3, with_line=False, ms=5.8),
        )
        ax1.set(
            xlim=(0.0, 4.0),
            ylim=(0.0, 1.02),
            xlabel=r"Normalized thickness $\tau b = (\beta / \gamma_0) b$",
            ylabel=r"Planar reduction factor $R_{\mathrm{2D}}$",
            title="(b) Universal Branch–Mihran curves",
        )
        figstyle.spectrum_grid(ax1)
        ax1.legend(loc="lower right", frameon=False, fontsize=8.5)
        figstyle.save(fig, fig_dir / "06_reduction_factor_vs_thickness.png", close=True, auto_legend_below=False)

        # ------------------------------------------------------------------
        # Fig 07: Reduction factor R(f) and beta_q(f) vs frequency (1-10 GHz)
        # ------------------------------------------------------------------
        fig, (ax0, ax1) = figstyle.new_figure(
            width_in=figstyle.FULL_WIDTH_IN,
            height_in=3.4,
            ncols=2,
            nrows=1,
            constrained_layout=False,
        )
        fig.subplots_adjust(left=0.09, right=0.98, bottom=0.18, top=0.88, wspace=0.34)

        f_sweep_dense = np.linspace(0.5e9, 10.5e9, 200)
        w_sweep_dense = 2.0 * np.pi * f_sweep_dense
        R_2d_f = reduction_factor_planar(
            w_sweep_dense, U0_REF, B_BEAM_REF, A_TUNNEL_REF
        )
        R_3d_cyl_f = reduction_factor_cylindrical(
            w_sweep_dense, U0_REF, B_BEAM_REF, A_TUNNEL_REF
        )
        R_3d_rect_f = reduction_factor_rectangular(
            w_sweep_dense, U0_REF, B_BEAM_REF, B_BEAM_REF, A_TUNNEL_REF, A_TUNNEL_REF
        )

        f_pts_ghz = kw_task["freqs"] * 1e-9
        ax0.plot(
            f_sweep_dense * 1e-9,
            R_2d_f,
            label="2D sheet (Branch–Mihran)",
            **figstyle.series_style(2, with_marker=False),
        )
        ax0.plot(
            f_pts_ghz,
            kw_task["R_fit"],
            label="Wakis 2D Bloch FIT",
            **figstyle.series_style(2, with_line=False, ms=6.0),
        )
        ax0.plot(
            f_sweep_dense * 1e-9,
            R_3d_rect_f,
            label="3D square tunnel (Theory)",
            **figstyle.series_style(0, with_marker=False),
        )
        ax0.plot(
            f_sweep_dense * 1e-9,
            R_3d_cyl_f,
            label="3D circular (Mealy 2022)",
            **figstyle.series_style(1, with_marker=False),
        )
        ax0.set(
            xlim=(0.5, 10.5),
            ylim=(0.0, 0.85),
            xlabel="Modulation frequency f [GHz]",
            ylabel=r"Reduction factor $R(f)$",
            title="(a) Reduction factor $R(f)$ (1–10 GHz)",
        )
        figstyle.spectrum_grid(ax0)
        ax0.legend(loc="upper left", frameon=False, fontsize=8.5)

        bq_2d_f = R_2d_f * (WP_REF / (GAMMA0_REF**1.5)) / U0_REF
        bq_fit = kw_task["omega_q"] / U0_REF
        ax1.plot(
            f_sweep_dense * 1e-9,
            bq_2d_f,
            label=r"Analytic $\beta_q(f)$",
            **figstyle.series_style(2, with_marker=False),
        )
        ax1.plot(
            f_pts_ghz,
            bq_fit,
            label=r"Wakis 2D Bloch FIT $\beta_q(f)$",
            **figstyle.series_style(1, with_line=False, ms=6.0),
        )
        ax1.set(
            xlim=(0.5, 10.5),
            xlabel="Modulation frequency f [GHz]",
            ylabel=r"Reduced wavenumber $\beta_q(f)$ [rad/m]",
            title=r"(b) Reduced wavenumber $\beta_q(f)$",
        )
        figstyle.spectrum_grid(ax1)
        ax1.legend(loc="upper left", frameon=False, fontsize=8.5)
        figstyle.save(fig, fig_dir / "07_reduction_factor_vs_frequency.png", close=True, auto_legend_below=False)

        # ------------------------------------------------------------------
        # Fig 08: 2D vs 3D transverse electric field fringing comparison
        # ------------------------------------------------------------------
        fig, (ax0, ax1) = figstyle.new_figure(
            width_in=figstyle.FULL_WIDTH_IN,
            height_in=3.5,
            ncols=2,
            nrows=1,
            constrained_layout=False,
        )
        fig.subplots_adjust(left=0.08, right=0.98, bottom=0.19, top=0.88, wspace=0.36)

        x3 = beam3d["x_nodes"] * 1e3
        y3 = beam3d["y_nodes"] * 1e3
        ez_3d = beam3d["Ez_xy"]
        ez_norm = ez_3d / max(float(np.max(np.abs(ez_3d))), 1e-30)

        im0 = ax0.pcolormesh(
            x3,
            y3,
            ez_norm.T,
            cmap=figstyle.CMAP_DIVERGING,
            vmin=-1.0,
            vmax=1.0,
            shading="gouraud",
        )
        fig.colorbar(im0, ax=ax0, label=r"$E_z(x, y) / E_{z,\max}$")
        schematic.overlay_geometry(
            ax0,
            s_3d,
            plane="xy",
            unit="mm",
            structures=structs_3d,
            probes=probes_3d,
            show_boundaries=True,
            show_materials=True,
        )
        _add_pec_wall_tags_2d(ax0, "+y: PEC", "−y: PEC")
        xc = 0.5 * (x3[:-1] + x3[1:])
        yc = 0.5 * (y3[:-1] + y3[1:])
        ex_c = 0.5 * (beam3d["Ex_xy"][:, :-1] + beam3d["Ex_xy"][:, 1:])
        ey_c = 0.5 * (beam3d["Ey_xy"][:-1, :] + beam3d["Ey_xy"][1:, :])
        step_q = max(1, len(xc) // 10)
        XC, YC = np.meshgrid(xc[::step_q], yc[::step_q], indexing="ij")
        ax0.quiver(
            XC,
            YC,
            ex_c[::step_q, ::step_q],
            ey_c[::step_q, ::step_q],
            color=figstyle.INK,
            alpha=0.72,
            zorder=7,
        )
        ax0.set(
            aspect="equal",
            xlabel="Transverse x [mm]",
            ylabel="Transverse y [mm]",
            title=r"(a) 3D tunnel $E_z(x,y)$ & $\mathbf{E}_\perp$",
        )

        labels_geom = [
            "1D\nSlab",
            "2D\nSheet",
            "3D\nSquare",
            "3D\nRound",
        ]
        r_vals = [
            1.0,
            reduction_factor_planar(OMEGA0_REF, U0_REF, B_BEAM_REF, A_TUNNEL_REF),
            reduction_factor_rectangular(
                OMEGA0_REF, U0_REF, B_BEAM_REF, B_BEAM_REF, A_TUNNEL_REF, A_TUNNEL_REF
            ),
            reduction_factor_cylindrical(OMEGA0_REF, U0_REF, B_BEAM_REF, A_TUNNEL_REF),
        ]
        colors_bar = [
            "#94A3B8",
            figstyle.PALETTE[2],
            figstyle.PALETTE[0],
            figstyle.PALETTE[1],
        ]
        bars = ax1.bar(labels_geom, r_vals, color=colors_bar, width=0.52, edgecolor=figstyle.INK)
        for bar, v in zip(bars, r_vals):
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                v + 0.025,
                f"R = {v:.3f}",
                ha="center",
                va="bottom",
                fontsize=8.6,
                fontweight="bold",
                color=figstyle.INK,
            )
        ax1.set(
            ylim=(0.0, 1.16),
            ylabel=r"Reduction factor $R$ ($b/a=0.25,\;5\;\mathrm{GHz}$)",
            title="(b) Wall-fringing: 1D vs 2D vs 3D",
        )
        figstyle.spectrum_grid(ax1)
        figstyle.save(fig, fig_dir / "08_2d_vs_3d_fringing_comparison.png", close=True, auto_legend_below=False)

        # ------------------------------------------------------------------
        # Fig 09: 2D spatial maps of rho1(z, y), J1z(z, y), and Ez(z, y)
        # ------------------------------------------------------------------
        fig, axes = figstyle.new_figure(
            width_in=figstyle.FULL_WIDTH_IN,
            height_in=5.8,
            ncols=1,
            nrows=3,
            sharex=True,
            constrained_layout=False,
        )
        fig.subplots_adjust(left=0.09, right=0.96, bottom=0.10, top=0.93, hspace=0.45)

        x_mm = bunch["x_coords"]
        y_mm = bunch["y_coords"]
        for ax, arr, lbl, title_str in (
            (
                axes[0],
                bunch["rho1_2d"],
                r"$\rho_1 / |\rho_1|_{\max}$",
                r"(a) Space-charge density perturbation $\rho_1(z, y, t)$",
            ),
            (
                axes[1],
                bunch["J1_2d"],
                r"$J_{1z} / |J_{1z}|_{\max}$",
                r"(b) Small-signal axial current density $J_{1z}(z, y, t)$",
            ),
            (
                axes[2],
                bunch["Ex_2d"],
                r"$E_z / |E_z|_{\max}$",
                r"(c) Space-charge axial electric field $E_z(z, y, t)$",
            ),
        ):
            amax = max(float(np.max(np.abs(arr))), 1e-30)
            im = ax.pcolormesh(
                x_mm,
                y_mm,
                (arr / amax).T,
                cmap=figstyle.CMAP_DIVERGING,
                vmin=-1.0,
                vmax=1.0,
                shading="auto",
            )
            fig.colorbar(im, ax=ax, label=lbl, pad=0.015)
            schematic.overlay_geometry(
                ax,
                s_bunch,
                plane="xy",
                unit="mm",
                structures=structs_bunch,
                sources=srcs_bunch,
                monitors=mons_bunch,
                show_boundaries=True,
                show_materials=True,
                show_sources=True,
                show_monitors=True,
            )
            _add_pec_wall_tags_2d(ax, "+y: PEC wall", "−y: PEC wall")
            ax.set(ylabel="y [mm]", title=title_str)
        axes[2].set_xlabel("Axial drift coordinate z [mm]")
        figstyle.save(fig, fig_dir / "09_modulated_beam_field_map.png", close=True, auto_legend_below=False)

        # ------------------------------------------------------------------
        # Fig 10: Klystron drift-space current & velocity standing-wave beat
        # ------------------------------------------------------------------
        fig, (ax0, ax1) = figstyle.new_figure(
            width_in=figstyle.FULL_WIDTH_IN,
            height_in=3.4,
            ncols=2,
            nrows=1,
            constrained_layout=False,
        )
        fig.subplots_adjust(left=0.09, right=0.98, bottom=0.18, top=0.88, wspace=0.34)

        z_mm = bunch["z_drift"] * 1e3
        ax0.plot(
            z_mm,
            bunch["I1_th_norm"],
            label=r"Analytic $|\sin(\beta_q z)|$",
            **figstyle.series_style(2, with_marker=False),
        )
        ax0.plot(
            z_mm[::3],
            bunch["I1_norm"][::3],
            label=f"FIT $|I_1|$ (L_beat={bunch['L_beat_fit']*1e3:.1f} mm)",
            **figstyle.series_style(1, with_line=False, ms=4.2),
        )
        l_q4_mm = 0.5 * bunch["L_beat_th"] * 1e3
        l_q2_mm = bunch["L_beat_th"] * 1e3
        ax0.axvline(
            l_q4_mm,
            color=figstyle.PALETTE[0],
            ls=":",
            lw=1.5,
            marker="none",
            label=f"λ_q/4 = {l_q4_mm:.1f} mm",
        )
        ax0.axvline(
            l_q2_mm,
            color=figstyle.INK,
            ls="--",
            lw=1.3,
            marker="none",
            label=f"λ_q/2 = {l_q2_mm:.1f} mm",
        )
        ax0.set(
            xlabel="Distance downstream of gap z [mm]",
            ylabel=r"Normalized $|I_1(z)| / I_{1,\max}$",
            ylim=(-0.03, 1.14),
            title="(a) Klystron current beat",
        )
        figstyle.spectrum_grid(ax0)
        figstyle.place_legend_below(ax0, ncol=1, y_offset=-0.22)

        ax1.plot(
            z_mm,
            bunch["I1_norm"],
            label=r"Current $|I_1(z)| \propto |\sin(\beta_q z)|$",
            **figstyle.series_style(1, with_marker=False),
        )
        ax1.plot(
            z_mm,
            bunch["V1_norm"],
            label=r"Voltage $|V_1(z)| \propto |\cos(\beta_q z)|$",
            **figstyle.series_style(0, with_marker=False),
        )
        ax1.plot(
            z_mm,
            bunch["V1_th_norm"],
            label=r"Analytic $|\cos(\beta_q z)|$",
            **figstyle.series_style(7, with_marker=False),
        )
        ax1.set(
            xlabel="Distance downstream of gap z [mm]",
            ylabel="Normalized envelope",
            ylim=(-0.03, 1.14),
            title=r"(b) $\pi/2$ quadrature ($V_1$ vs $I_1$)",
        )
        figstyle.spectrum_grid(ax1)
        figstyle.place_legend_below(ax1, ncol=1, y_offset=-0.22)
        figstyle.save(fig, fig_dir / "10_klystron_bunching_beat.png", close=True, auto_legend_below=False)

        # ------------------------------------------------------------------
        # Fig 11: Animated time-domain space-charge wave propagation GIF
        # ------------------------------------------------------------------
        if "movie_rho" in bunch and len(bunch["movie_rho"]) > 0:
            frames_rho = bunch["movie_rho"]
            times_ns = bunch["movie_times"] * 1e9
            rmax = max(float(np.max(np.abs(frames_rho))), 1e-30)
            pil_frames = []
            for k in range(len(frames_rho)):
                fig_g, ax_g = plt.subplots(figsize=(6.8, 3.0), dpi=130)
                im_g = ax_g.pcolormesh(
                    x_mm,
                    y_mm,
                    (frames_rho[k] / rmax).T,
                    cmap=figstyle.CMAP_DIVERGING,
                    vmin=-1.0,
                    vmax=1.0,
                    shading="auto",
                )
                fig_g.colorbar(im_g, ax=ax_g, label=r"$\rho_1(z,y,t)/\rho_{1,\max}$")
                schematic.overlay_geometry(
                    ax_g,
                    s_bunch,
                    plane="xy",
                    unit="mm",
                    structures=structs_bunch,
                    sources=srcs_bunch,
                    monitors=mons_bunch,
                    show_boundaries=True,
                    show_materials=True,
                    show_sources=True,
                    show_monitors=True,
                )
                _add_pec_wall_tags_2d(ax_g, "+y: PEC wall", "−y: PEC wall")
                ax_g.set(
                    xlabel="Axial drift coordinate z [mm]",
                    ylabel="Transverse y [mm]",
                    title=f"Space-charge wave bunching propagation (t = {times_ns[k]:.2f} ns)",
                )
                figstyle.finish(fig_g, auto_legend_below=False)
                fig_g.canvas.draw()
                rgba = np.asarray(fig_g.canvas.buffer_rgba())
                pil_frames.append(Image.fromarray(rgba).convert("P", palette=Image.ADAPTIVE))
                plt.close(fig_g)

            if pil_frames:
                pil_frames[0].save(
                    fig_dir / "11_space_charge_wave_propagation.gif",
                    save_all=True,
                    append_images=pil_frames[1:],
                    duration=90,
                    loop=0,
                )

        # ------------------------------------------------------------------
        # Fig 12: Chu kinetic power sign (fast vs slow wave) & energy balance
        # ------------------------------------------------------------------
        fig, (ax0, ax1) = figstyle.new_figure(
            width_in=figstyle.FULL_WIDTH_IN,
            height_in=3.4,
            ncols=2,
            nrows=1,
            constrained_layout=False,
        )
        fig.subplots_adjust(left=0.09, right=0.98, bottom=0.18, top=0.88, wspace=0.34)

        s_chu = chu["s_mm"]
        ax0.plot(
            s_chu,
            chu["Pk_fast_z"] * 1e3,
            label=f"Fast wave: ⟨P_k⟩ = +{chu['Pk_fast_mean']*1e3:.3f} mW",
            **figstyle.series_style(2, with_marker=False),
        )
        ax0.plot(
            s_chu,
            chu["Pk_slow_z"] * 1e3,
            label=f"Slow wave: ⟨P_k⟩ = {chu['Pk_slow_mean']*1e3:.3f} mW",
            **figstyle.series_style(1, with_marker=False),
        )
        ax0.fill_between(s_chu, 0.0, chu["Pk_fast_z"] * 1e3, color=figstyle.PALETTE[2], alpha=0.15)
        ax0.fill_between(s_chu, 0.0, chu["Pk_slow_z"] * 1e3, color=figstyle.PALETTE[1], alpha=0.15)
        ax0.axhline(0.0, color=figstyle.INK, ls="--", lw=1.1, marker="none")
        ax0.set(
            xlabel="Axial coordinate z [mm]",
            ylabel=r"Chu kinetic power $P_k(z)$ [mW]",
            title="(a) Fast (P_k > 0) vs slow (P_k < 0)",
        )
        figstyle.spectrum_grid(ax0)
        ax0.legend(loc="upper right", frameon=False, fontsize=8.5)

        t_chu = chu["t_ns"]
        u_tot0 = float(np.mean(chu["u_tot"]))
        ax1.plot(
            t_chu,
            chu["u_kin"] / u_tot0,
            label=r"Kinetic $U_{\mathrm{kin}}(t) / U_{\mathrm{tot}}$",
            **figstyle.series_style(0, with_marker=False),
        )
        ax1.plot(
            t_chu,
            chu["u_em"] / u_tot0,
            label=r"EM $U_{\mathrm{EM}}(t) / U_{\mathrm{tot}}$",
            **figstyle.series_style(1, with_marker=False),
        )
        ax1.plot(
            t_chu,
            chu["u_tot"] / u_tot0,
            label=f"Total U_tot (drift {chu['rel_energy_drift']*100:.4f}%)",
            **figstyle.series_style(2, with_marker=False),
        )
        ax1.set(
            xlabel="Time t [ns]",
            ylabel="Normalized energy",
            ylim=(-0.05, 1.15),
            title="(b) Chu–Poynting conservation",
        )
        figstyle.spectrum_grid(ax1)
        ax1.legend(loc="center right", frameon=False, fontsize=8.5)
        figstyle.save(fig, fig_dir / "12_chu_kinetic_power_and_energy.png", close=True, auto_legend_below=False)

        # ------------------------------------------------------------------
        # Fig 13: Mesh-refinement convergence study (coarse, medium, fine)
        # ------------------------------------------------------------------
        fig, (ax0, ax1) = figstyle.new_figure(
            width_in=figstyle.FULL_WIDTH_IN,
            height_in=3.4,
            ncols=2,
            nrows=1,
            constrained_layout=False,
        )
        fig.subplots_adjust(left=0.09, right=0.98, bottom=0.18, top=0.88, wspace=0.34)

        levels = ["coarse", "medium", "fine"]
        n_b_arr = np.array(
            [float(tasks_data[f"disp_res_{lvl}"]["cells_across_feature"]) for lvl in levels]
        )
        r_fit_arr = np.array(
            [float(tasks_data[f"disp_res_{lvl}"]["R_fit"]) for lvl in levels]
        )
        r_th_val = float(primary["R_th"])
        ratio_sq = (n_b_arr[-1] / n_b_arr[-2]) ** 2
        r_cont_ref = r_fit_arr[-1] + (r_fit_arr[-1] - r_fit_arr[-2]) / (ratio_sq - 1.0)
        rel_err_arr = np.maximum(np.abs(r_fit_arr - r_cont_ref) / r_cont_ref, 1e-5)
        rel_err_bm = np.maximum(np.abs(r_fit_arr - r_th_val) / r_th_val, 1e-5)

        ax0.loglog(
            n_b_arr,
            rel_err_arr,
            label=rf"$|R_{{\mathrm{{FIT}}}} - R_\infty| / R_\infty$ ($R_\infty={r_cont_ref:.4f}$)",
            **figstyle.series_style(2),
        )
        ax0.loglog(
            n_b_arr,
            rel_err_bm,
            label=rf"$|R_{{\mathrm{{FIT}}}} - R_{{\mathrm{{BM}}}}| / R_{{\mathrm{{BM}}}}$ ($R_{{\mathrm{{BM}}}}={r_th_val:.4f}$)",
            **figstyle.series_style(0),
        )
        guide_2nd = rel_err_arr[0] * (n_b_arr[0] / n_b_arr) ** 2
        ax0.loglog(
            n_b_arr,
            guide_2nd,
            label=r"$\mathcal{O}(\Delta x^2)$ slope",
            **figstyle.series_style(1, with_marker=False),
        )
        ax0.set_xticks(n_b_arr)
        ax0.xaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))
        ax0.xaxis.set_minor_formatter(mticker.NullFormatter())
        ax0.set(
            xlabel="Cells across beam half-thickness b",
            ylabel="Relative error in R",
            title="(a) Mesh convergence of R",
        )
        figstyle.spectrum_grid(ax0)
        ax0.legend(loc="lower left", frameon=False, fontsize=8.5)

        for idx_l, lvl in enumerate(levels):
            d = tasks_data[f"disp_res_{lvl}"]
            ax1.plot(
                d["fft_freqs"] * 1e-9,
                d["fft_spec"],
                label=f"{lvl.capitalize()} (N_b={d['cells_across_feature']:.0f}, R={d['R_fit']:.4f})",
                **figstyle.series_style(idx_l, with_marker=False),
            )
        ax1.axvline(f_slow_th_ghz, color=figstyle.INK, ls=":", lw=1.2, marker="none", label="Analytic f₋, f₊")
        ax1.axvline(f_fast_th_ghz, color=figstyle.INK, ls=":", lw=1.2, marker="none")
        ax1.set(
            xlim=(4.3, 5.7),
            xlabel="Frequency f [GHz]",
            ylabel="Normalized |E(f)|",
            title=f"(b) Spectral doublet (R_BM={r_th_val:.4f})",
        )
        figstyle.spectrum_grid(ax1)
        ax1.legend(loc="upper center", frameon=False, fontsize=8.5)
        figstyle.save(fig, fig_dir / "13_resolution_convergence.png", close=True, auto_legend_below=False)

        # ------------------------------------------------------------------
        # Fig 14: Linearized fluid validity limits vs nonlinear PIC/Bessel bunching
        # ------------------------------------------------------------------
        fig, (ax0, ax1) = figstyle.new_figure(
            width_in=figstyle.FULL_WIDTH_IN,
            height_in=3.4,
            ncols=2,
            nrows=1,
            constrained_layout=False,
        )
        fig.subplots_adjust(left=0.09, right=0.98, bottom=0.18, top=0.88, wspace=0.34)

        X_arr = chu["X_arr"]
        ax0.plot(
            X_arr,
            chu["I1_linear"],
            label=r"Linear fluid $|I_1|/I_0 = X$",
            **figstyle.series_style(0, with_marker=False),
        )
        ax0.plot(
            X_arr,
            chu["I1_bessel"],
            label=r"Bessel $|I_1|/I_0 = 2|J_1(X)|$",
            **figstyle.series_style(2, with_marker=False),
        )
        ax0.plot(
            X_arr,
            chu["I2_bessel"],
            label=r"2nd harm. $2|J_2(2X)|$ (PIC)",
            **figstyle.series_style(1, with_marker=False),
        )
        ax0.plot(
            X_arr,
            chu["I3_bessel"],
            label=r"3rd harm. $2|J_3(3X)|$ (PIC)",
            **figstyle.series_style(4, with_marker=False),
        )
        ax0.axvspan(
            0.0,
            0.5,
            color="#D1FAE5",
            alpha=0.55,
            label="Linear regime (X ≤ 0.5)",
        )
        ax0.axvline(
            1.0,
            color=figstyle.INK,
            ls=":",
            lw=1.5,
            marker="none",
            label="Overtaking (X = 1.0)",
        )
        ax0.set(
            xlim=(0.0, 3.0),
            ylim=(0.0, 1.55),
            xlabel=r"Bunching parameter $X$",
            ylabel=r"Harmonic current $|I_n(X)| / I_0$",
            title="(a) Linear vs Bessel bunching",
        )
        figstyle.spectrum_grid(ax0)
        figstyle.place_legend_below(ax0, ncol=2, y_offset=-0.24)

        phi0 = np.linspace(-np.pi, np.pi, 400)
        for idx_x, (X_val, lbl) in enumerate(
            (
                (0.25, "X = 0.25 (linear fluid valid)"),
                (1.00, "X = 1.00 (trajectory crossing)"),
                (1.60, "X = 1.60 (multi-stream, PIC)"),
            )
        ):
            psi = phi0 - X_val * np.sin(phi0)
            v_norm = 1.0 + 0.15 * X_val * np.sin(phi0)
            ax1.plot(
                psi / np.pi,
                v_norm,
                label=lbl,
                **figstyle.series_style(idx_x, with_marker=False),
            )
        ax1.set(
            xlim=(-1.05, 1.05),
            xlabel=r"Arrival phase $(\omega t - \beta_0 z)/\pi$",
            ylabel=r"Velocity $v_z / u_0$",
            title=r"(b) Phase-space fold at $X \geq 1$",
        )
        figstyle.spectrum_grid(ax1)
        figstyle.place_legend_below(ax1, ncol=1, y_offset=-0.24)
        figstyle.save(fig, fig_dir / "14_linear_vs_nonlinear_bunching_limits.png", close=True, auto_legend_below=False)

    print(f"Rendered 00a, 00b, and 14 figures in {fig_dir}")
