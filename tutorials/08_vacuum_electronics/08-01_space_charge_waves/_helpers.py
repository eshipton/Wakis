# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""
Physics simulation runners and plotting helpers for Tutorial 8.1:
Space-Charge Waves on a Drifting Electron Beam (`wakis.beam_fluid.LinearizedBeam`).
"""

from pathlib import Path

import matplotlib.animation as animation
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from scipy.constants import c as c_light, e as q_e, epsilon_0 as eps_0, m_e
from scipy.optimize import brentq
from scipy.special import jn as bessel_jn

import wakis
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
# Figure generation suite (14 figures + animated GIF)
# ---------------------------------------------------------------------------


def render_all_figures(tasks_data, fig_dir):
    """
    Render all 14 diagnostic and validation figures for Tutorial 8.1 into
    ``fig_dir`` from the loaded ``tasks_data`` dictionary.
    """
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    primary = tasks_data["disp_res_fine"]
    kw_task = tasks_data["kw_and_freq_sweep"]
    thick_task = tasks_data["thickness_sweep"]
    beam3d = tasks_data["beam_3d_tunnel"]
    bunch = tasks_data["drift_tube_bunching"]
    chu = tasks_data["chu_energy_and_slow_wave"]

    # ------------------------------------------------------------------
    # Fig 01: Drift-tube & velocity-modulating gap schematic layout
    # ------------------------------------------------------------------
    fig, (ax0, ax1) = plt.subplots(
        1, 2, figsize=(11.2, 3.9), dpi=150, gridspec_kw={"width_ratios": [1.85, 1.0]}
    )
    L_mm = 115.0
    a_mm = A_TUNNEL_REF * 1e3
    b_mm = 0.8
    ax0.add_patch(
        mpatches.Rectangle(
            (0.0, -a_mm), L_mm, 2.0 * a_mm, facecolor="#f4f6f9", edgecolor="none"
        )
    )
    ax0.add_patch(
        mpatches.Rectangle(
            (0.0, -b_mm),
            L_mm - 8.0,
            2.0 * b_mm,
            facecolor="#2b5c8f",
            alpha=0.28,
            edgecolor="#2b5c8f",
            lw=1.3,
            label=r"Drifting electron beam ($|y|\leq b$, $\mathbf{u}_0 = u_0\hat{\mathbf{z}}$)",
        )
    )
    # Inflow & outflow CPML slabs
    ax0.axvspan(0.0, 3.0, color="#7570b3", alpha=0.22, label="Inflow / Outflow CPML")
    ax0.axvspan(L_mm - 3.0, L_mm, color="#7570b3", alpha=0.22)
    ax0.axvspan(
        L_mm - 8.0,
        L_mm - 3.0,
        color="#e6ab02",
        alpha=0.25,
        label=r"Smooth $\rho_0(z)\to 0$ taper before CPML",
    )
    # Modulating grid gap
    ax0.axvline(5.0, color="#d95f02", lw=2.0, ls="--", label="Modulating grid gap $d_g$")
    ax0.axvline(6.8, color="#d95f02", lw=2.0, ls="--")
    # PEC tunnel walls
    ax0.axhline(+a_mm, color="0.15", lw=3.0, label=r"Grounded PEC drift tube ($|y|=a$)")
    ax0.axhline(-a_mm, color="0.15", lw=3.0)
    ax0.annotate(
        r"$\mathbf{u}_0 = 0.16c$" + "\n" + r"($V_0 = 6.7\;\mathrm{kV}$)",
        xy=(28.0, 0.0),
        xytext=(14.0, 0.0),
        arrowprops=dict(arrowstyle="->", lw=1.8, color="#1b9e77"),
        fontsize=8.5,
        va="center",
        color="#1b9e77",
        weight="bold",
    )
    ax0.annotate(
        r"RF gap $V_m\sin(\omega t)$" + "\n" + r"($v_1 \neq 0,\;\rho_1 = 0$)",
        xy=(6.0, b_mm),
        xytext=(12.0, 1.45),
        arrowprops=dict(arrowstyle="->", lw=1.2, color="#d95f02"),
        fontsize=8.0,
        color="#d95f02",
    )
    ax0.annotate(
        r"Space-charge bunches $\lambda_q/2 = \pi/\beta_q$",
        xy=(55.0, 0.45),
        xytext=(42.0, 1.45),
        arrowprops=dict(arrowstyle="->", lw=1.1, color="#2b5c8f"),
        fontsize=8.0,
        color="#2b5c8f",
    )
    ax0.set(
        xlim=(0.0, L_mm),
        ylim=(-a_mm * 1.18, a_mm * 1.18),
        xlabel="Axial drift coordinate z [mm]",
        ylabel="Transverse coordinate y [mm]",
        title="(a) Longitudinal drift-tube schematic (Mealy et al. 2022)",
    )
    ax0.legend(loc="lower right", fontsize=7.3, ncol=2)

    # Transverse cross section (2D slab vs 3D square/circular tunnel)
    ax1.add_patch(
        mpatches.Rectangle(
            (-a_mm, -a_mm),
            2.0 * a_mm,
            2.0 * a_mm,
            facecolor="#f4f6f9",
            edgecolor="0.15",
            lw=2.5,
            label="3D square PEC tunnel ($2a \\times 2a$)",
        )
    )
    circ_t = mpatches.Circle(
        (0.0, 0.0),
        a_mm,
        fill=False,
        edgecolor="0.45",
        ls="--",
        lw=1.5,
        label="Circular tunnel ($R_t = a$)",
    )
    ax1.add_patch(circ_t)
    ax1.add_patch(
        mpatches.Rectangle(
            (-a_mm, -0.5),
            2.0 * a_mm,
            1.0,
            facecolor="#1b9e77",
            alpha=0.18,
            edgecolor="#1b9e77",
            ls=":",
            lw=1.4,
            label="2D planar sheet beam ($|y|\\leq b$)",
        )
    )
    circ_b = mpatches.Circle(
        (0.0, 0.0),
        0.5,
        facecolor="#2b5c8f",
        alpha=0.45,
        edgecolor="#2b5c8f",
        lw=1.6,
        label="3D pencil beam ($R_b = 0.5$ mm)",
    )
    ax1.add_patch(circ_b)
    ax1.set(
        xlim=(-a_mm * 1.2, a_mm * 1.2),
        ylim=(-a_mm * 1.2, a_mm * 1.2),
        aspect="equal",
        xlabel="Transverse x [mm]",
        ylabel="Transverse y [mm]",
        title="(b) Transverse cross-section ($a=2.0$ mm, $b=0.5$ mm)",
    )
    ax1.grid(True, ls=":", alpha=0.4)
    ax1.legend(loc="upper right", fontsize=7.2)
    fig.tight_layout()
    fig.savefig(fig_dir / "01_drift_tube_geometry.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Fig 02: Mesh zoom & transverse/axial profile weights
    # ------------------------------------------------------------------
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.0, 3.9), dpi=150)
    for lvl, col, mk, ls in (
        ("coarse", "#7570b3", "s", "--"),
        ("medium", "#d95f02", "^", "-."),
        ("fine", "#1b9e77", "o", "-"),
    ):
        d = tasks_data[f"disp_res_{lvl}"]
        y_mm = d["y_nodes"] * 1e3
        w_y = d["w_perp_y"]
        ax0.plot(
            y_mm,
            w_y,
            marker=mk,
            ls=ls,
            color=col,
            lw=1.6,
            ms=4.5,
            label=f"{lvl.capitalize()} ($N_y={int(d['ny'])}$, $N_b={d['cells_across_feature']:.0f}$ cells/$b$)",
        )
    y_ref = np.linspace(-2.0, 2.0, 500)
    ax0.plot(
        y_ref,
        (np.abs(y_ref) <= 0.5).astype(float),
        color="0.2",
        lw=1.2,
        ls=":",
        label="Exact top-hat $|y|\\leq 0.5$ mm",
    )
    ax0.set(
        xlim=(-1.2, 1.2),
        ylim=(-0.05, 1.12),
        xlabel="Transverse coordinate y [mm]",
        ylabel=r"Dual-cell overlap weight $w_\perp(y) = \rho_0(y)/\rho_{0,\mathrm{peak}}$",
        title="(a) Subcell transverse beam weight across $|y|=b$",
    )
    ax0.grid(True, ls=":", alpha=0.4)
    ax0.legend(loc="upper right", fontsize=8.0)

    s_mm = bunch["s_coords_mm"]
    w_ax = bunch["w_axial"]
    ax1.plot(
        s_mm,
        w_ax,
        "o-",
        color="#2b5c8f",
        lw=1.8,
        ms=3.2,
        label=r"Axial beam weight $w_s(z)$ ($\sin(\pi\xi/2)$ ramp)",
    )
    ax1.axvspan(
        s_mm[-1],
        115.0,
        color="#7570b3",
        alpha=0.25,
        label="Outflow CPML ($\\rho_0 = 0$ identically)",
    )
    ax1.axvspan(
        s_mm[-10],
        s_mm[-1],
        color="#e6ab02",
        alpha=0.22,
        label="10-cell smooth density taper before CPML",
    )
    ax1.set(
        xlim=(s_mm[-35], 115.0),
        ylim=(-0.05, 1.12),
        xlabel="Axial coordinate z near collector [mm]",
        ylabel=r"Normalized DC density $\rho_0(z)/\rho_{0,\mathrm{peak}}$",
        title="(b) Outflow density ramp preventing PML beam instability",
    )
    ax1.grid(True, ls=":", alpha=0.4)
    ax1.legend(loc="lower left", fontsize=8.0)
    fig.tight_layout()
    fig.savefig(fig_dir / "02_mesh_and_profile_zoom.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Fig 03: Transverse TM space-charge wave eigenmode profiles
    # ------------------------------------------------------------------
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.0, 4.0), dpi=150)
    y_n_mm = primary["y_nodes"] * 1e3
    y_e_mm = primary["y_edges"] * 1e3
    y_dense = np.linspace(-A_TUNNEL_REF, A_TUNNEL_REF, 400)
    Es_d, Ey_d, J1_d, _, _, _ = planar_tm_eigenmode(
        y_dense, OMEGA0_REF, U0_REF, B_BEAM_REF, A_TUNNEL_REF
    )

    ax0.plot(
        y_dense * 1e3,
        Es_d,
        color="#2b5c8f",
        lw=2.2,
        label=r"Analytic $E_z(y)$ ($\cos(Ty)$ / $\sinh(\tau(a-|y|))$)",
    )
    ax0.plot(
        y_n_mm,
        primary["Ex_fit"],
        "o",
        color="#d95f02",
        ms=4.2,
        label="Wakis 2D FIT $E_z(y)$",
    )
    ax0.plot(
        y_dense * 1e3,
        J1_d,
        "--",
        color="#1b9e77",
        lw=1.8,
        label=r"Analytic $J_{1z}(y)/J_{\max}$",
    )
    ax0.plot(
        y_n_mm,
        primary["J1_fit"],
        "s",
        color="#1b9e77",
        ms=3.8,
        alpha=0.8,
        label="Wakis 2D FIT $J_{1z}(y)/J_{\\max}$",
    )
    ax0.axvspan(-0.5, 0.5, color="#2b5c8f", alpha=0.10, label="Beam core ($|y|\\leq 0.5$ mm)")
    ax0.set(
        xlabel="Transverse coordinate y [mm]",
        ylabel="Normalized amplitude",
        title=r"(a) Axial electric field $E_z(y)$ & beam current $J_{1z}(y)$",
    )
    ax0.grid(True, ls=":", alpha=0.4)
    ax0.legend(loc="upper right", fontsize=7.8)

    ax1.plot(
        y_dense * 1e3,
        Ey_d,
        color="#7570b3",
        lw=2.2,
        label=r"Analytic $E_y(y)/E_z(0)$",
    )
    ax1.plot(
        y_e_mm,
        primary["Ey_fit"],
        "o",
        color="#d95f02",
        ms=4.2,
        label="Wakis 2D FIT $E_y(y)/E_z(0)$",
    )
    ax1.axvspan(-0.5, 0.5, color="#2b5c8f", alpha=0.10, label="Beam core ($|y|\\leq 0.5$ mm)")
    ax1.axhline(0.0, color="0.4", ls=":", lw=1.0)
    ax1.set(
        xlabel="Transverse coordinate y [mm]",
        ylabel=r"Normalized transverse field $E_y(y)/E_z(0)$",
        title=r"(b) Transverse fringing electric field $E_y(y)$",
    )
    ax1.grid(True, ls=":", alpha=0.4)
    ax1.legend(loc="lower right", fontsize=8.0)
    fig.tight_layout()
    fig.savefig(
        fig_dir / "03_transverse_tm_eigenmodes.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig)

    # ------------------------------------------------------------------
    # Fig 04: Space-charge wave k-omega dispersion diagram
    # ------------------------------------------------------------------
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.2, 4.1), dpi=150)
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
        "--",
        color="0.35",
        lw=1.6,
        label=r"Synchronous beam line $\omega = \beta u_0$",
    )
    ax0.plot(
        beta_dense,
        f_fast_d,
        color="#1b9e77",
        lw=2.1,
        label=r"Fast space-charge wave $\omega_+ = \beta u_0 + \omega_q$ (Theory)",
    )
    ax0.plot(
        beta_dense,
        f_slow_d,
        color="#d95f02",
        lw=2.1,
        label=r"Slow space-charge wave $\omega_- = \beta u_0 - \omega_q$ (Theory)",
    )
    ax0.plot(
        betas_fit,
        f_fast_fit,
        "o",
        color="#1b9e77",
        ms=6.0,
        label=r"Wakis Bloch FIT $\omega_+(\beta)$",
    )
    ax0.plot(
        betas_fit,
        f_slow_fit,
        "s",
        color="#d95f02",
        ms=6.0,
        label=r"Wakis Bloch FIT $\omega_-(\beta)$",
    )
    ax0.plot(
        beta_dense,
        f_light_d,
        ":",
        color="#7570b3",
        lw=1.5,
        label=r"Light line $\omega = c\beta$",
    )
    ax0.set(
        xlim=(50.0, 1400.0),
        ylim=(0.0, 11.5),
        xlabel=r"Axial wavenumber $\beta$ [rad/m]",
        ylabel="Frequency $f = \\omega / 2\\pi$ [GHz]",
        title=r"(a) Space-charge wave $(\beta, \omega)$ dispersion diagram",
    )
    ax0.grid(True, ls=":", alpha=0.4)
    ax0.legend(loc="upper left", fontsize=7.6)

    # Beam-frame frequency shift (omega - beta u0) / (2 pi) vs beta
    df_th = (w_fast_d - beta_dense * U0_REF) / (2.0 * np.pi) * 1e-3  # MHz
    df_fast_fit = (kw_task["omega_fast"] - betas_fit * U0_REF) / (2.0 * np.pi) * 1e-3
    df_slow_fit = (kw_task["omega_slow"] - betas_fit * U0_REF) / (2.0 * np.pi) * 1e-3
    fp_rel_mhz = (WP_REF / (GAMMA0_REF**1.5)) / (2.0 * np.pi) * 1e-3

    ax1.plot(
        beta_dense,
        +df_th,
        color="#1b9e77",
        lw=2.1,
        label=r"$+\omega_q(\beta)/2\pi$ (Fast wave, Theory)",
    )
    ax1.plot(
        beta_dense,
        -df_th,
        color="#d95f02",
        lw=2.1,
        label=r"$-\omega_q(\beta)/2\pi$ (Slow wave, Theory)",
    )
    ax1.plot(
        betas_fit,
        df_fast_fit,
        "o",
        color="#1b9e77",
        ms=6.0,
        label="Wakis FIT fast shift",
    )
    ax1.plot(
        betas_fit,
        df_slow_fit,
        "s",
        color="#d95f02",
        ms=6.0,
        label="Wakis FIT slow shift",
    )
    ax1.axhline(0.0, color="0.4", ls="--", lw=1.2)
    ax1.axhline(
        +fp_rel_mhz,
        color="0.5",
        ls=":",
        lw=1.3,
        label=r"1D unreduced limit $\pm \omega_p / (2\pi\gamma_0^{3/2})$",
    )
    ax1.axhline(-fp_rel_mhz, color="0.5", ls=":", lw=1.3)
    ax1.set(
        xlim=(50.0, 1400.0),
        xlabel=r"Axial wavenumber $\beta$ [rad/m]",
        ylabel=r"Doppler shift $(\omega_\pm - \beta u_0)/2\pi$ [MHz]",
        title=r"(b) Co-moving frame splitting $\pm \omega_q(\beta) = \pm R(\beta)\omega_p/\gamma_0^{3/2}$",
    )
    ax1.grid(True, ls=":", alpha=0.4)
    ax1.legend(loc="best", fontsize=7.6)
    fig.tight_layout()
    fig.savefig(fig_dir / "04_kw_dispersion_diagram.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Fig 05: Time-domain beating trace & Doppler doublet FFT spectrum
    # ------------------------------------------------------------------
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.0, 3.9), dpi=150)
    t_ns = primary["trace_t"] * 1e9
    ex_re = np.real(primary["trace_Ex"])
    ex_env = np.abs(primary["trace_Ex"])
    scale_e = max(float(np.max(ex_env)), 1e-30)
    ax0.plot(
        t_ns,
        ex_re / scale_e,
        color="#2b5c8f",
        lw=0.9,
        alpha=0.85,
        label=r"$\mathrm{Re}[E_z(t)] / E_{\max}$",
    )
    ax0.plot(
        t_ns,
        ex_env / scale_e,
        color="#d95f02",
        lw=2.0,
        label=r"Beating envelope $|E_z(t)| \propto |\sin(\omega_q t)|$",
    )
    ax0.plot(t_ns, -ex_env / scale_e, color="#d95f02", lw=2.0)
    ax0.set(
        xlabel="Time t [ns]",
        ylabel="Normalized axial field",
        title=r"(a) Fast/slow wave temporal beating at $f_0 = 5.0\;\mathrm{GHz}$",
    )
    ax0.grid(True, ls=":", alpha=0.4)
    ax0.legend(loc="upper right", fontsize=8.0)

    f_ghz = primary["fft_freqs"] * 1e-9
    spec_db = 20.0 * np.log10(np.maximum(primary["fft_spec"], 1e-6))
    f_slow_th_ghz = primary["omega_slow_th"] / (2.0 * np.pi) * 1e-9
    f_fast_th_ghz = primary["omega_fast_th"] / (2.0 * np.pi) * 1e-9

    ax1.plot(
        f_ghz, spec_db, color="#2b5c8f", lw=2.0, label="Wakis Bloch FIT spectrum"
    )
    ax1.axvline(
        F0_REF * 1e-9,
        color="0.4",
        ls="--",
        lw=1.4,
        label=f"Beam carrier $f_0 = \\beta u_0 / 2\\pi = {F0_REF * 1e-9:.2f}$ GHz",
    )
    ax1.axvline(
        f_slow_th_ghz,
        color="#d95f02",
        ls=":",
        lw=1.8,
        label=f"Slow wave $f_- = {f_slow_th_ghz:.3f}$ GHz (Theory)",
    )
    ax1.axvline(
        f_fast_th_ghz,
        color="#1b9e77",
        ls=":",
        lw=1.8,
        label=f"Fast wave $f_+ = {f_fast_th_ghz:.3f}$ GHz (Theory)",
    )
    ax1.set(
        xlim=(3.8, 6.2),
        ylim=(-55.0, 3.0),
        xlabel="Frequency f [GHz]",
        ylabel="Normalized spectral magnitude [dB]",
        title=r"(b) Space-charge wave Doppler doublet $\omega_\pm = \beta u_0 \pm \omega_q$",
    )
    ax1.grid(True, ls=":", alpha=0.4)
    ax1.legend(loc="lower center", fontsize=7.8)
    fig.tight_layout()
    fig.savefig(
        fig_dir / "05_doppler_splitting_spectra.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig)

    # ------------------------------------------------------------------
    # Fig 06: Reduction factor R vs beam thickness / filling ratio b/a
    # ------------------------------------------------------------------
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.2, 4.0), dpi=150)
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
        color="#1b9e77",
        lw=2.2,
        label="2D planar sheet beam (Branch & Mihran 1955)",
    )
    ax0.plot(
        thick_task["fill_ratios"],
        thick_task["R_fit"],
        "o",
        color="#1b9e77",
        ms=6.5,
        label="Wakis 2D FIT ($N_z=1$, sheet beam)",
    )
    ax0.plot(
        fill_dense,
        R_3d_rect_curve,
        "--",
        color="#2b5c8f",
        lw=2.0,
        label="3D square beam in square tunnel (Theory)",
    )
    ax0.plot(
        [beam3d["b_half"] / beam3d["a_half"]],
        [beam3d["R_3d_fit"]],
        "D",
        color="#2b5c8f",
        ms=7.0,
        label="Wakis 3D FIT (square tunnel)",
    )
    ax0.plot(
        fill_dense,
        R_3d_cyl_curve,
        "-.",
        color="#d95f02",
        lw=2.0,
        label="3D pencil beam in circular tube (Mealy et al. 2022, Fig. 3b)",
    )
    ax0.set(
        xlim=(0.0, 1.05),
        ylim=(0.0, 0.85),
        xlabel=r"Tunnel filling ratio $b / a$ ($a = 2.0\;\mathrm{mm}$, $f_0 = 5\;\mathrm{GHz}$)",
        ylabel=r"Plasma-frequency reduction factor $R = \omega_q \gamma_0^{3/2} / \omega_p$",
        title="(a) Reduction factor $R$ vs beam filling ratio $b / a$",
    )
    ax0.grid(True, ls=":", alpha=0.4)
    ax0.legend(loc="upper left", fontsize=7.5)

    # Universal planar Branch & Mihran (1955) curves vs normalized thickness tau * b
    tau_b_grid = np.linspace(0.05, 4.0, 200)
    u0_bm = U0_REF
    b_bm = 1.0e-3
    _, g_bm = relativistic_gamma(u0=u0_bm)
    omega_bm = tau_b_grid * g_bm * u0_bm / b_bm
    for ratio_ab, col, ls in (
        (1.0, "#7570b3", ":"),
        (1.5, "#d95f02", "-."),
        (2.0, "#2b5c8f", "--"),
        (4.0, "#1b9e77", "-"),
        (np.inf, "0.2", "-"),
    ):
        R_bm = reduction_factor_planar(omega_bm, u0_bm, b_bm, a=ratio_ab * b_bm)
        lbl = (
            r"$a/b = \infty$ (free space)"
            if np.isinf(ratio_ab)
            else f"$a/b = {ratio_ab:g}$"
        )
        ax1.plot(tau_b_grid, R_bm, color=col, ls=ls, lw=1.9, label=lbl)

    # Overlay Wakis FIT points from frequency sweep (where a/b = 4.0)
    tau_b_fit = (kw_task["betas"] / GAMMA0_REF) * B_BEAM_REF
    ax1.plot(
        tau_b_fit,
        kw_task["R_fit"],
        "o",
        color="#1b9e77",
        ms=6.0,
        label="Wakis 2D FIT ($a/b = 4$)",
    )
    ax1.set(
        xlim=(0.0, 4.0),
        ylim=(0.0, 1.02),
        xlabel=r"Normalized transverse wavenumber $\tau b = (\beta / \gamma_0) b$",
        ylabel=r"Planar reduction factor $R_{\mathrm{2D}}(\tau b, a/b)$",
        title="(b) Universal planar Branch & Mihran (1955) curves",
    )
    ax1.grid(True, ls=":", alpha=0.4)
    ax1.legend(loc="lower right", fontsize=7.6)
    fig.tight_layout()
    fig.savefig(
        fig_dir / "06_reduction_factor_vs_thickness.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig)

    # ------------------------------------------------------------------
    # Fig 07: Reduction factor R(f) and beta_q(f) vs frequency (1-10 GHz)
    # ------------------------------------------------------------------
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.2, 4.0), dpi=150)
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
        color="#1b9e77",
        lw=2.2,
        label="2D sheet beam exact (Branch & Mihran 1955)",
    )
    ax0.plot(
        f_pts_ghz,
        kw_task["R_fit"],
        "o",
        color="#1b9e77",
        ms=6.5,
        label="Wakis 2D Bloch FIT",
    )
    ax0.plot(
        f_sweep_dense * 1e-9,
        R_3d_rect_f,
        "--",
        color="#2b5c8f",
        lw=1.8,
        label="3D square beam in square tunnel (Theory)",
    )
    ax0.plot(
        f_sweep_dense * 1e-9,
        R_3d_cyl_f,
        color="#d95f02",
        lw=2.1,
        label="3D cylindrical exact (Mealy et al. 2022, Fig. 3a)",
    )
    ax0.set(
        xlim=(0.5, 10.5),
        ylim=(0.0, 0.85),
        xlabel="Modulation frequency f [GHz]",
        ylabel=r"Reduction factor $R(f)$ ($b=0.5\;\mathrm{mm},\;a=2.0\;\mathrm{mm}$)",
        title="(a) Reduction factor $R(f)$ across 1–10 GHz",
    )
    ax0.grid(True, ls=":", alpha=0.4)
    ax0.legend(loc="upper left", fontsize=7.6)

    bq_2d_f = R_2d_f * (WP_REF / (GAMMA0_REF**1.5)) / U0_REF
    bq_fit = kw_task["omega_q"] / U0_REF
    ax1.plot(
        f_sweep_dense * 1e-9,
        bq_2d_f,
        color="#1b9e77",
        lw=2.2,
        label=r"Analytic $\beta_q(f) = R(f)\omega_p / (\gamma_0^{3/2} u_0)$",
    )
    ax1.plot(
        f_pts_ghz,
        bq_fit,
        "o",
        color="#d95f02",
        ms=6.5,
        label=r"Wakis 2D Bloch FIT $\beta_q(f)$",
    )
    ax1.set(
        xlim=(0.5, 10.5),
        xlabel="Modulation frequency f [GHz]",
        ylabel=r"Reduced plasma wavenumber $\beta_q(f)$ [rad/m]",
        title=r"(b) Reduced space-charge wavenumber $\beta_q(f)$ vs frequency",
    )
    ax1.grid(True, ls=":", alpha=0.4)
    ax1.legend(loc="upper left", fontsize=8.0)
    fig.tight_layout()
    fig.savefig(
        fig_dir / "07_reduction_factor_vs_frequency.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig)

    # ------------------------------------------------------------------
    # Fig 08: 2D vs 3D transverse electric field fringing comparison
    # ------------------------------------------------------------------
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.0, 4.2), dpi=150)
    x3 = beam3d["x_nodes"] * 1e3
    y3 = beam3d["y_nodes"] * 1e3
    ez_3d = beam3d["Ez_xy"]
    ez_norm = ez_3d / max(float(np.max(np.abs(ez_3d))), 1e-30)

    im0 = ax0.pcolormesh(
        x3, y3, ez_norm.T, cmap="RdBu_r", vmin=-1.0, vmax=1.0, shading="gouraud"
    )
    fig.colorbar(im0, ax=ax0, label=r"Normalized $E_z(x, y) / E_{z,\max}$")
    b3_mm = beam3d["b_half"] * 1e3
    ax0.add_patch(
        mpatches.Rectangle(
            (-b3_mm, -b3_mm),
            2.0 * b3_mm,
            2.0 * b3_mm,
            fill=False,
            edgecolor="k",
            ls="--",
            lw=1.6,
            label=f"3D beam boundary ($|x|,|y|\\leq {b3_mm:.1f}$ mm)",
        )
    )
    # Quiver of transverse field (Ex, Ey) at cell centers
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
        color="0.15",
        alpha=0.75,
    )
    ax0.set(
        aspect="equal",
        xlabel="Transverse x [mm]",
        ylabel="Transverse y [mm]",
        title=r"(a) 3D tunnel $E_z(x,y)$ & radial $\mathbf{E}_\perp$ fringing",
    )
    ax0.legend(loc="upper right", fontsize=7.8)

    # Bar / curve comparison of 1D vs 2D vs 3D reduction factors at b = 0.5 mm, a = 2.0 mm
    labels_geom = [
        "1D Infinite\nSlab ($R=1$)",
        "2D Planar\nSheet Beam",
        "3D Square\nBeam/Tunnel",
        "3D Cylindrical\nPencil Beam",
    ]
    r_vals = [
        1.0,
        reduction_factor_planar(OMEGA0_REF, U0_REF, B_BEAM_REF, A_TUNNEL_REF),
        reduction_factor_rectangular(
            OMEGA0_REF, U0_REF, B_BEAM_REF, B_BEAM_REF, A_TUNNEL_REF, A_TUNNEL_REF
        ),
        reduction_factor_cylindrical(OMEGA0_REF, U0_REF, B_BEAM_REF, A_TUNNEL_REF),
    ]
    colors_bar = ["0.65", "#1b9e77", "#2b5c8f", "#d95f02"]
    bars = ax1.bar(labels_geom, r_vals, color=colors_bar, width=0.52, edgecolor="0.2")
    for bar, v in zip(bars, r_vals):
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            v + 0.025,
            f"R = {v:.3f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            weight="bold",
        )
    ax1.set(
        ylim=(0.0, 1.15),
        ylabel=r"Reduction factor $R$ ($b=0.5\;\mathrm{mm},\;a=2.0\;\mathrm{mm},\;5\;\mathrm{GHz}$)",
        title="(b) Dimensional wall-fringing reduction: 1D vs 2D vs 3D",
    )
    ax1.grid(True, axis="y", ls=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(
        fig_dir / "08_2d_vs_3d_fringing_comparison.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig)

    # ------------------------------------------------------------------
    # Fig 09: 2D spatial maps of rho1(z, y), J1z(z, y), and Ez(z, y)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(10.6, 6.8), dpi=150, sharex=True)
    x_mm = bunch["x_coords"]
    y_mm = bunch["y_coords"]
    for ax, arr, lbl, title_str in (
        (
            axes[0],
            bunch["rho1_2d"],
            r"$\rho_1 / |\rho_1|_{\max}$",
            r"(a) Instantaneous space-charge density perturbation $\rho_1(z, y, t)$",
        ),
        (
            axes[1],
            bunch["J1_2d"],
            r"$J_{1z} / |J_{1z}|_{\max}$",
            r"(b) Instantaneous small-signal axial beam current density $J_{1z}(z, y, t)$",
        ),
        (
            axes[2],
            bunch["Ex_2d"],
            r"$E_z / |E_z|_{\max}$",
            r"(c) Self-consistent space-charge axial electric field $E_z(z, y, t)$",
        ),
    ):
        amax = max(float(np.max(np.abs(arr))), 1e-30)
        im = ax.pcolormesh(
            x_mm,
            y_mm,
            (arr / amax).T,
            cmap="RdBu_r",
            vmin=-1.0,
            vmax=1.0,
            shading="auto",
        )
        fig.colorbar(im, ax=ax, label=lbl, pad=0.015)
        ax.axhline(+0.8, color="k", ls="--", lw=0.9, alpha=0.6)
        ax.axhline(-0.8, color="k", ls="--", lw=0.9, alpha=0.6)
        ax.set(ylabel="y [mm]", title=title_str)
    axes[2].set_xlabel("Axial drift coordinate z [mm]")
    fig.tight_layout()
    fig.savefig(
        fig_dir / "09_modulated_beam_field_map.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig)

    # ------------------------------------------------------------------
    # Fig 10: Klystron drift-space current & velocity standing-wave beat
    # ------------------------------------------------------------------
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.2, 4.0), dpi=150)
    z_mm = bunch["z_drift"] * 1e3
    ax0.plot(
        z_mm,
        bunch["I1_th_norm"],
        color="#1b9e77",
        lw=2.2,
        label=r"Analytic $|I_1(z)|/I_{\max} = |\sin(\beta_q z)|$ (Ramo 1939; Mealy et al. Fig. 1b)",
    )
    ax0.plot(
        z_mm,
        bunch["I1_norm"],
        "o",
        color="#d95f02",
        ms=3.6,
        alpha=0.85,
        label=f"Wakis FIT $|I_1(z)|$ ($L_{{\\mathrm{{beat}}}}={bunch['L_beat_fit']*1e3:.2f}$ mm vs {bunch['L_beat_th']*1e3:.2f} mm)",
    )
    # Mark quarter-wavelength klystron bunching peak and half-wavelength node
    l_q4_mm = 0.5 * bunch["L_beat_th"] * 1e3
    l_q2_mm = bunch["L_beat_th"] * 1e3
    ax0.axvline(
        l_q4_mm,
        color="#2b5c8f",
        ls=":",
        lw=1.5,
        label=f"Optimum klystron bunching $\\lambda_q/4 = {l_q4_mm:.1f}$ mm",
    )
    ax0.axvline(
        l_q2_mm,
        color="0.4",
        ls="--",
        lw=1.4,
        label=f"First current node $\\lambda_q/2 = {l_q2_mm:.1f}$ mm",
    )
    ax0.set(
        xlabel="Distance downstream of modulating gap z [mm]",
        ylabel=r"Normalized AC current envelope $|I_1(z)| / I_{1,\max}$",
        ylim=(-0.03, 1.12),
        title="(a) Modulated beam current beat (Mealy et al. 2022, Fig. 1b)",
    )
    ax0.grid(True, ls=":", alpha=0.4)
    ax0.legend(loc="upper right", fontsize=7.4)

    ax1.plot(
        z_mm,
        bunch["I1_norm"],
        color="#d95f02",
        lw=2.0,
        label=r"AC beam current envelope $|I_1(z)| \propto |\sin(\beta_q z)|$",
    )
    ax1.plot(
        z_mm,
        bunch["V1_norm"],
        color="#2b5c8f",
        lw=2.0,
        ls="--",
        label=r"Kinetic voltage envelope $|V_1(z)| \propto |\cos(\beta_q z)|$",
    )
    ax1.plot(
        z_mm,
        bunch["V1_th_norm"],
        ":",
        color="0.3",
        lw=1.4,
        label=r"Analytic $|\cos(\beta_q z)|$",
    )
    ax1.set(
        xlabel="Distance downstream of modulating gap z [mm]",
        ylabel="Normalized standing-wave envelope",
        ylim=(-0.03, 1.12),
        title=r"(b) $\pi/2$ spatial quadrature between $V_1(z)$ and $I_1(z)$",
    )
    ax1.grid(True, ls=":", alpha=0.4)
    ax1.legend(loc="upper right", fontsize=7.8)
    fig.tight_layout()
    fig.savefig(fig_dir / "10_klystron_bunching_beat.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Fig 11: Animated time-domain space-charge wave propagation GIF
    # ------------------------------------------------------------------
    if "movie_rho" in bunch and len(bunch["movie_rho"]) > 0:
        frames_rho = bunch["movie_rho"]
        times_ns = bunch["movie_times"] * 1e9
        rmax = max(float(np.max(np.abs(frames_rho))), 1e-30)

        fig, ax = plt.subplots(figsize=(8.8, 3.2), dpi=110)
        im = ax.pcolormesh(
            x_mm,
            y_mm,
            (frames_rho[0] / rmax).T,
            cmap="RdBu_r",
            vmin=-1.0,
            vmax=1.0,
            shading="auto",
        )
        fig.colorbar(im, ax=ax, label=r"Normalized $\rho_1(z,y,t)/\rho_{1,\max}$")
        ax.axhline(+0.8, color="k", ls="--", lw=0.9, alpha=0.6)
        ax.axhline(-0.8, color="k", ls="--", lw=0.9, alpha=0.6)
        title_obj = ax.set_title(
            f"Space-charge wave bunching propagation (t = {times_ns[0]:.2f} ns)"
        )
        ax.set_xlabel("Axial drift coordinate z [mm]")
        ax.set_ylabel("Transverse y [mm]")
        fig.tight_layout()

        def _update_frame(k):
            im.set_array((frames_rho[k] / rmax).T.ravel())
            title_obj.set_text(
                f"Space-charge wave bunching propagation (t = {times_ns[k]:.2f} ns)"
            )
            return (im, title_obj)

        ani = animation.FuncAnimation(
            fig, _update_frame, frames=len(frames_rho), interval=90, blit=False
        )
        ani.save(
            fig_dir / "11_space_charge_wave_propagation.gif",
            writer="pillow",
            fps=11,
        )
        plt.close(fig)

    # ------------------------------------------------------------------
    # Fig 12: Chu kinetic power sign (fast vs slow wave) & energy balance
    # ------------------------------------------------------------------
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.2, 4.0), dpi=150)
    s_chu = chu["s_mm"]
    ax0.plot(
        s_chu,
        chu["Pk_fast_z"] * 1e3,
        color="#1b9e77",
        lw=2.2,
        label=f"Fast wave ($\\omega_+ = \\beta u_0 + \\omega_q$): $\\langle P_k \\rangle = +{chu['Pk_fast_mean']*1e3:.3f}$ mW ($>0$)",
    )
    ax0.plot(
        s_chu,
        chu["Pk_slow_z"] * 1e3,
        color="#d95f02",
        lw=2.2,
        label=f"Slow wave ($\\omega_- = \\beta u_0 - \\omega_q$): $\\langle P_k \\rangle = {chu['Pk_slow_mean']*1e3:.3f}$ mW ($<0$)",
    )
    ax0.fill_between(s_chu, 0.0, chu["Pk_fast_z"] * 1e3, color="#1b9e77", alpha=0.15)
    ax0.fill_between(s_chu, 0.0, chu["Pk_slow_z"] * 1e3, color="#d95f02", alpha=0.15)
    ax0.axhline(0.0, color="0.3", ls="--", lw=1.2)
    ax0.set(
        xlabel="Axial coordinate z across one wavelength [mm]",
        ylabel=r"Instantaneous Chu kinetic power $P_k(z) = \int V_1 J_{1z}\,dA$ [mW]",
        title="(a) Positive-energy fast wave vs negative-energy slow wave",
    )
    ax0.grid(True, ls=":", alpha=0.4)
    ax0.legend(loc="upper right", fontsize=7.6)

    t_chu = chu["t_ns"]
    u_tot0 = float(np.mean(chu["u_tot"]))
    ax1.plot(
        t_chu,
        chu["u_kin"] / u_tot0,
        color="#2b5c8f",
        lw=1.8,
        label=r"Chu kinetic energy $U_{\mathrm{kin}}(t) / U_{\mathrm{tot}}$",
    )
    ax1.plot(
        t_chu,
        chu["u_em"] / u_tot0,
        color="#d95f02",
        lw=1.8,
        label=r"Electromagnetic energy $U_{\mathrm{EM}}(t) / U_{\mathrm{tot}}$",
    )
    ax1.plot(
        t_chu,
        chu["u_tot"] / u_tot0,
        color="#1b9e77",
        lw=2.2,
        label=f"Total Chu-Poynting $U_{{\\mathrm{{tot}}}}(t)$ (drift = {chu['rel_energy_drift']*100:.4f}%)",
    )
    ax1.set(
        xlabel="Time t [ns]",
        ylabel="Normalized energy",
        ylim=(-0.05, 1.15),
        title="(b) Coupled EM + beam Chu–Poynting energy conservation",
    )
    ax1.grid(True, ls=":", alpha=0.4)
    ax1.legend(loc="center right", fontsize=7.8)
    fig.tight_layout()
    fig.savefig(
        fig_dir / "12_chu_kinetic_power_and_energy.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig)

    # ------------------------------------------------------------------
    # Fig 13: Mesh-refinement convergence study (coarse, medium, fine)
    # ------------------------------------------------------------------
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.0, 4.0), dpi=150)
    levels = ["coarse", "medium", "fine"]
    n_b_arr = np.array(
        [float(tasks_data[f"disp_res_{lvl}"]["cells_across_feature"]) for lvl in levels]
    )
    r_fit_arr = np.array(
        [float(tasks_data[f"disp_res_{lvl}"]["R_fit"]) for lvl in levels]
    )
    r_th_val = float(primary["R_th"])
    # Second-order Richardson extrapolation from medium -> fine for the
    # continuum full-Maxwell reference limit
    ratio_sq = (n_b_arr[-1] / n_b_arr[-2]) ** 2
    r_cont_ref = r_fit_arr[-1] + (r_fit_arr[-1] - r_fit_arr[-2]) / (ratio_sq - 1.0)
    rel_err_arr = np.maximum(np.abs(r_fit_arr - r_cont_ref) / r_cont_ref, 1e-5)
    rel_err_bm = np.maximum(np.abs(r_fit_arr - r_th_val) / r_th_val, 1e-5)

    ax0.loglog(
        n_b_arr,
        rel_err_arr,
        "o-",
        color="#1b9e77",
        lw=2.2,
        ms=7.0,
        label=rf"Wakis FIT $|R_{{\mathrm{{FIT}}}} - R_\infty| / R_\infty$ ($R_\infty={r_cont_ref:.4f}$)",
    )
    ax0.loglog(
        n_b_arr,
        rel_err_bm,
        "s:",
        color="#2b5c8f",
        lw=1.8,
        ms=6.0,
        label=rf"Wakis FIT $|R_{{\mathrm{{FIT}}}} - R_{{\mathrm{{BM}}}}| / R_{{\mathrm{{BM}}}}$ ($R_{{\mathrm{{BM}}}}={r_th_val:.4f}$)",
    )
    guide_2nd = rel_err_arr[0] * (n_b_arr[0] / n_b_arr) ** 2
    ax0.loglog(
        n_b_arr,
        guide_2nd,
        "--",
        color="#d95f02",
        lw=1.6,
        label=r"Second-order $\mathcal{O}(\Delta x^2)$ reference slope",
    )
    ax0.set_xticks(n_b_arr)
    ax0.xaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))
    ax0.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax0.set(
        xlabel="Cells across beam half-thickness b",
        ylabel="Relative error in reduction factor R",
        title="(a) Mesh convergence of reduction factor $R$",
    )
    ax0.grid(True, which="both", ls=":", alpha=0.4)
    ax0.legend(loc="lower left", fontsize=7.4)

    for lvl, col, ls in (
        ("coarse", "#7570b3", ":"),
        ("medium", "#d95f02", "--"),
        ("fine", "#1b9e77", "-"),
    ):
        d = tasks_data[f"disp_res_{lvl}"]
        ax1.plot(
            d["fft_freqs"] * 1e-9,
            d["fft_spec"],
            color=col,
            ls=ls,
            lw=1.8,
            label=f"{lvl.capitalize()} ($N_b={d['cells_across_feature']:.0f}$, $R={d['R_fit']:.4f}$)",
        )
    ax1.axvline(f_slow_th_ghz, color="0.3", ls="--", lw=1.2, label="Analytic $f_-, f_+$")
    ax1.axvline(f_fast_th_ghz, color="0.3", ls="--", lw=1.2)
    ax1.set(
        xlim=(4.3, 5.7),
        xlabel="Frequency f [GHz]",
        ylabel="Normalized spectral magnitude",
        title=f"(b) Spectral doublet across resolutions ($R_{{\\mathrm{{BM}}}}={r_th_val:.4f}$)",
    )
    ax1.grid(True, ls=":", alpha=0.4)
    ax1.legend(loc="upper center", fontsize=7.6)
    fig.tight_layout()
    fig.savefig(fig_dir / "13_resolution_convergence.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Fig 14: Linearized fluid validity limits vs nonlinear PIC/Bessel bunching
    # ------------------------------------------------------------------
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.2, 4.0), dpi=150)
    X_arr = chu["X_arr"]
    ax0.plot(
        X_arr,
        chu["I1_linear"],
        "--",
        color="#2b5c8f",
        lw=2.0,
        label=r"Linearized cold fluid $|I_1|/I_0 = X$",
    )
    ax0.plot(
        X_arr,
        chu["I1_bessel"],
        color="#1b9e77",
        lw=2.2,
        label=r"Nonlinear ballistic fundamental $|I_1|/I_0 = 2|J_1(X)|$",
    )
    ax0.plot(
        X_arr,
        chu["I2_bessel"],
        color="#d95f02",
        lw=1.8,
        label=r"Second harmonic $|I_2|/I_0 = 2|J_2(2X)|$ (needs PIC)",
    )
    ax0.plot(
        X_arr,
        chu["I3_bessel"],
        color="#7570b3",
        lw=1.6,
        ls="-.",
        label=r"Third harmonic $|I_3|/I_0 = 2|J_3(3X)|$ (needs PIC)",
    )
    ax0.axvspan(
        0.0,
        0.5,
        color="#1b9e77",
        alpha=0.12,
        label="Small-signal linear regime ($X \\lesssim 0.5$)",
    )
    ax0.axvline(
        1.0,
        color="#e7298a",
        ls=":",
        lw=1.7,
        label="Trajectory overtaking / wave breaking ($X = 1.0$)",
    )
    ax0.axvline(
        1.841,
        color="0.35",
        ls="--",
        lw=1.3,
        label="Bessel saturation peak ($X = 1.841$)",
    )
    ax0.set(
        xlim=(0.0, 3.0),
        ylim=(0.0, 1.55),
        xlabel=r"Bunching parameter $X = (\beta_0 V_m M / 2V_0)\,\sin(\beta_q z)/(\beta_q z_0)$",
        ylabel=r"Harmonic current ratio $|I_n(X)| / I_0$",
        title="(a) Linear cold-fluid vs nonlinear Bessel harmonic bunching",
    )
    ax0.grid(True, ls=":", alpha=0.4)
    ax0.legend(loc="upper left", fontsize=7.2)

    # Phase-space (omega t_0 vs arrival phasepsi = omega t - beta_0 z = omega t_0 - X sin(omega t_0))
    phi0 = np.linspace(-np.pi, np.pi, 400)
    for X_val, col, ls, lbl in (
        (0.25, "#1b9e77", "-", "$X = 0.25$ (single-valued, linear fluid valid)"),
        (1.00, "#d95f02", "--", "$X = 1.00$ (onset of trajectory crossing, $\\rho_1 \\to \\infty$)"),
        (1.60, "#e7298a", "-.", "$X = 1.60$ (multi-streaming / overtaking, PIC required)"),
    ):
        psi = phi0 - X_val * np.sin(phi0)
        v_norm = 1.0 + 0.15 * X_val * np.sin(phi0)
        ax1.plot(psi / np.pi, v_norm, color=col, ls=ls, lw=2.0, label=lbl)
    ax1.set(
        xlim=(-1.05, 1.05),
        xlabel=r"Normalized arrival phase $(\omega t - \beta_0 z)/\pi$",
        ylabel=r"Normalized electron velocity $v_z / u_0$",
        title=r"(b) Phase-space $(z, v_z)$ fold & multi-streaming at $X \geq 1$",
    )
    ax1.grid(True, ls=":", alpha=0.4)
    ax1.legend(loc="upper left", fontsize=7.6)
    fig.tight_layout()
    fig.savefig(
        fig_dir / "14_linear_vs_nonlinear_bunching_limits.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(f"Rendered 14 figures in {fig_dir}")
