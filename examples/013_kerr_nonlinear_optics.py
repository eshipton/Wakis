# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Example 013 — Kerr (chi^(3)) Nonlinear Optics in FIT: SPM, Maker-Fringe THG,
Cavity Resonance Shift, and Lugiato–Lefever Soliton Microcombs
============================================================================

Demonstrates time-domain simulation of third-order (Kerr, ``chi^(3)``) optical
nonlinearities in `SolverFIT3D` using `KerrMaterial` (`wakis.nonlinear`),
following Taflove & Hagness (2005), Ch. 9 [1]; Joseph & Taflove, *IEEE Trans.
Antennas Propag.* **45**(3), 364–374 (1997) [2]; and Herr et al., *Nat.
Photonics* **8**(2), 145–152 (2014) [3].

Benchmarks and outputs
----------------------
1. **Self-phase modulation (SPM)**: Continuous-wave propagation at ``f_0 = 200 THz``
   (``lambda_0 = 1.5 um``) through an index-matched Kerr slab of length
   ``L = 3.0 um`` (``eps_inf = 2.25``, ``n_0 = 1.5``, ``chi^(3) = 2.4e-19 m^2/V^2``,
   ``dz = 15 nm = lambda / 66.7``). Verifies ``Delta Phi_NL = n_2 k_0 I L`` with
   ``n_2 = (3 / 4) chi^(3) / (c eps_0 n_0^2)`` across all three solver update
   schemes (``'newton'``, ``'fixed_point'``, ``'explicit_ade'``).
2. **Third-harmonic generation (THG) Maker fringes**: Propagation at ``f_0 = 150 THz``
   (``lambda_0 = 2.0 um``, third harmonic ``3 f_0 = 450 THz``, ``dz = 15 nm =
   lambda_3 / 27.2``) in a dispersive Kerr slab with coherence length
   ``L_c = pi / |Delta k| = 5.291 um``, verifying ``eta_THG(L) ~ L^2 sinc^2(Delta k L / 2)``.
3. **Kerr cavity resonance shift**: Free ringdown (`wakis.harminv`) of a 1D
   Fabry–Pérot cavity (``V_eff,NL = (2/3) V_geom``) and a 2D TM cavity
   (``V_eff,NL = (4/9) V_geom``) vs first-order perturbation theory
   ``Delta f_NL / f_0 = -(c n_2 / n_0^2) (U / V_eff,NL)``.
4. **Lugiato–Lefever dissipative Kerr soliton (DKS) comb**: Split-step Fourier
   integration (`solve_lle`) vs the analytical ``sech^2`` spectral envelope for
   ``d_2 = 0.01`` and ``d_2 = 0.04``, verifying ``Delta mu_FWHM ~ 1 / sqrt(D_2)``.

Saves all numerical results to ``examples/data/013_kerr_nonlinear_optics.json``
and two publication-style figures to ``docs/img/``:
  1. ``docs/img/kerr_spm_and_thg.png``
  2. ``docs/img/kerr_cavity_shift_and_lle.png``

References
----------
[1] A. Taflove and S. C. Hagness, *Computational Electrodynamics: The
    Finite-Difference Time-Domain Method*, 3rd ed., Artech House, Ch. 9 (2005).
[2] R. M. Joseph and A. Taflove, "FDTD Maxwell's equations models for nonlinear
    electrodynamics and optics," *IEEE Trans. Antennas Propag.* 45(3), 364 (1997),
    doi:10.1109/8.558652.
[3] T. Herr et al., "Temporal solitons in optical microresonators," *Nat. Photonics*
    8(2), 145 (2014), doi:10.1038/nphoton.2013.343.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c, epsilon_0, mu_0

import wakis
from wakis.estimate import estimate, preflight
from wakis.figstyle import PALETTE, apply_style, panel_label, save, spectrum_grid
from wakis.materials import Lorentz
from wakis.nonlinear import (
    KerrMaterial,
    chi3_to_n2,
    effective_nonlinear_volume,
    kerr_cavity_frequency_shift,
    lle_soliton_analytic,
    optical_intensity,
    solve_lle,
    spm_phase_shift,
    thg_conversion_efficiency,
)
from wakis.sources import ContinuousSource, CurrentSource

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "examples" / "data" / "013_kerr_nonlinear_optics.json"
IMG_DIR = REPO_ROOT / "docs" / "img"


def run_spm_benchmark():
    """Run 1D FIT self-phase modulation sweep across intensities and update methods."""
    t0 = time.perf_counter()
    f0 = 200.0e12
    lam0 = c / f0
    omega0 = 2.0 * np.pi * f0
    eps_inf = 2.25
    n0 = float(np.sqrt(eps_inf))
    chi3 = 2.4e-19
    n2 = float(chi3_to_n2(chi3, eps_r=eps_inf))
    L = 3.0e-6
    dz = 15.0e-9  # 66.7 cells/lambda in n0 = 1.5
    steps_per_period = 240
    dt = 1.0 / (steps_per_period * f0)
    Z_med = float(np.sqrt(mu_0 / (epsilon_0 * eps_inf)))

    grid = wakis.GridFIT3D(
        -0.5e-6,
        0.5e-6,
        -0.5e-6,
        0.5e-6,
        -1.5e-6,
        4.5e-6,
        2,
        2,
        int(6.0e-6 / dz),
        verbose=0,
    )
    n_avg = 6 * steps_per_period
    Nt = 28 * steps_per_period

    est = estimate((grid.Nx, grid.Ny, grid.Nz), steps=Nt, cpml=True)
    preflight(est, "laptop")

    def _run_single(chi3_val, E0_target, method="newton"):
        solver = wakis.SolverFIT3D(
            grid,
            dt=dt,
            bc_low=["periodic", "periodic", "cpml"],
            bc_high=["periodic", "periodic", "cpml"],
            n_pml=12,
            bg=[eps_inf, 1.0, 0.0],
            verbose=0,
        )
        if chi3_val is not None:
            mat = KerrMaterial(eps_inf=eps_inf, chi3=chi3_val, method=method)
            solver.add_nonlinear_material(
                mat, center=(0.0, 0.0, 0.5 * L), size=(2.0e-6, 2.0e-6, L)
            )
        src = ContinuousSource(frequency=f0, ramp_time=4.0 / f0)
        sheet = CurrentSource(
            src,
            component="x",
            center=(0.0, 0.0, -0.5e-6),
            size=(1.0e-6, 1.0e-6, 0.0),
            amplitude=2.0 * E0_target / Z_med,
        )
        k_in = int(np.argmin(np.abs(solver.z - (-0.1e-6))))
        k_out = int(np.argmin(np.abs(solver.z - (L + 0.2e-6))))
        stride_z = solver.Nx * solver.Ny
        idx_in = k_in * stride_z
        idx_out = k_out * stride_z
        c1_in = 0.0j
        c1_out = 0.0j
        for n in range(Nt):
            sheet.update(solver, n * dt)
            solver.one_step()
            if n >= Nt - n_avg:
                t_e = (n + 1) * dt
                phase = np.exp(-1j * omega0 * t_e)
                c1_in += solver.E.array[idx_in] * phase
                c1_out += solver.E.array[idx_out] * phase
        c1_in *= 2.0 / n_avg
        c1_out *= 2.0 / n_avg
        return c1_in, c1_out, solver.E.array.copy()

    E_list = [3.0e7, 6.0e7, 9.0e7, 1.2e8, 1.5e8]
    c1_in_ref, c1_out_ref, E_lin_arr = _run_single(None, E_list[0])
    _, _, E_zero_arr = _run_single(0.0, E_list[0])
    zero_chi3_max_diff = float(np.max(np.abs(E_lin_arr - E_zero_arr)))

    I_list = []
    dphi_newton = []
    dphi_an = []
    dphi_wrong_factor = []
    rel_errors = []

    for E0 in E_list:
        scale = E0 / E_list[0]
        c1_in_lin = c1_in_ref * scale
        c1_out_lin = c1_out_ref * scale
        _, c1_out_nl, _ = _run_single(chi3, E0, method="newton")
        I_in = float(optical_intensity(np.abs(c1_in_lin), n0=n0))
        dphi_m = float(np.angle(c1_out_lin / c1_out_nl))
        dphi_a = float(spm_phase_shift(I_in, L, lam0, chi3=chi3, n0=n0))
        dphi_w = float(spm_phase_shift(I_in, L, lam0, chi3=chi3, n0=n0, factor=1.0))
        I_list.append(I_in)
        dphi_newton.append(dphi_m)
        dphi_an.append(dphi_a)
        dphi_wrong_factor.append(dphi_w)
        rel_errors.append(abs(dphi_m - dphi_a) / abs(dphi_a))

    # Compare fixed_point and explicit_ade at E0 = 9.0e7 V/m
    E_mid = 9.0e7
    scale_mid = E_mid / E_list[0]
    c1_out_lin_mid = c1_out_ref * scale_mid
    _, c1_fp, _ = _run_single(chi3, E_mid, method="fixed_point")
    _, c1_ade, _ = _run_single(chi3, E_mid, method="explicit_ade")
    dphi_fp = float(np.angle(c1_out_lin_mid / c1_fp))
    dphi_ade = float(np.angle(c1_out_lin_mid / c1_ade))

    return {
        "f0_hz": f0,
        "lambda0_m": lam0,
        "eps_inf": eps_inf,
        "n0": n0,
        "chi3_m2_v2": chi3,
        "n2_m2_w": n2,
        "length_m": L,
        "dz_m": dz,
        "cells_per_wavelength": (lam0 / n0) / dz,
        "E0_v_m": E_list,
        "intensity_w_m2": I_list,
        "dphi_newton_rad": dphi_newton,
        "dphi_analytic_rad": dphi_an,
        "dphi_wrong_factor1_rad": dphi_wrong_factor,
        "rel_errors": rel_errors,
        "max_rel_error": float(np.max(rel_errors)),
        "dphi_fixed_point_mid_rad": dphi_fp,
        "dphi_explicit_ade_mid_rad": dphi_ade,
        "zero_chi3_max_diff": zero_chi3_max_diff,
        "wall_time_s": float(time.perf_counter() - t0),
    }


def run_thg_maker_fringe_benchmark():
    """Run 1D FIT third-harmonic generation Maker-fringe benchmark."""
    t0 = time.perf_counter()
    f0 = 150.0e12
    lam0 = c / f0
    omega0 = 2.0 * np.pi * f0
    dz = 15.0e-9  # 27.2 cells/lambda at 3*f0 = 450 THz in n3 = 1.63
    steps_per_period = 300
    dt = 1.0 / (steps_per_period * f0)
    chi3 = 2.0e-19
    pole = Lorentz(delta_eps=0.45, omega_0=2.0 * np.pi * 800.0e12, gamma=0.0)
    mat_pm = KerrMaterial(eps_inf=2.00, chi3=chi3, poles=[pole], method="newton")
    eps1 = float(np.real(mat_pm.epsilon_discrete(f0, dt)))
    eps3 = float(np.real(mat_pm.epsilon_discrete(3.0 * f0, dt)))
    n1 = float(np.sqrt(eps1))
    n3 = float(np.sqrt(eps3))

    def k_disc(freq, eps_r):
        w = 2.0 * np.pi * freq
        s = (np.sqrt(eps_r) * dz / (c * dt)) * np.sin(0.5 * w * dt)
        return float((2.0 / dz) * np.arcsin(s))

    dk_disc = k_disc(3.0 * f0, eps3) - 3.0 * k_disc(f0, eps1)
    Lc = float(np.pi / dk_disc)

    Z1 = float(np.sqrt(mu_0 / (epsilon_0 * eps1)))
    E0 = 4.0e7
    grid = wakis.GridFIT3D(
        -0.5e-6,
        0.5e-6,
        -0.5e-6,
        0.5e-6,
        -1.5e-6,
        13.5e-6,
        2,
        2,
        int(15.0e-6 / dz),
        verbose=0,
    )
    n_avg = 6 * steps_per_period
    Nt = 40 * steps_per_period

    est = estimate((grid.Nx, grid.Ny, grid.Nz), steps=Nt, cpml=True)
    preflight(est, "laptop")

    solver = wakis.SolverFIT3D(
        grid,
        dt=dt,
        bc_low=["periodic", "periodic", "cpml"],
        bc_high=["periodic", "periodic", "cpml"],
        n_pml=12,
        bg=[eps1, 1.0, 0.0],
        verbose=0,
    )
    L_slab = 12.0e-6
    solver.add_nonlinear_material(
        mat_pm, center=(0.0, 0.0, 0.5 * L_slab), size=(2.0e-6, 2.0e-6, L_slab)
    )
    src = ContinuousSource(frequency=f0, ramp_time=4.0 / f0)
    sheet = CurrentSource(
        src,
        component="x",
        center=(0.0, 0.0, -0.5e-6),
        size=(1.0e-6, 1.0e-6, 0.0),
        amplitude=2.0 * E0 / Z1,
    )
    stride_z = solver.Nx * solver.Ny
    z_idx = np.arange(solver.Nz) * stride_z
    c1 = np.zeros(solver.Nz, dtype=complex)
    c3 = np.zeros(solver.Nz, dtype=complex)
    for n in range(Nt):
        sheet.update(solver, n * dt)
        solver.one_step()
        if n >= Nt - n_avg:
            t_e = (n + 1) * dt
            ex = solver.E.array[z_idx]
            c1 += ex * np.exp(-1j * omega0 * t_e)
            c3 += ex * np.exp(-1j * 3.0 * omega0 * t_e)
    c1 *= 2.0 / n_avg
    c3 *= 2.0 / n_avg

    k_in = int(np.argmin(np.abs(solver.z - 0.15e-6)))
    I1 = float(optical_intensity(np.abs(c1[k_in]), n0=n1))

    mask_curve = (solver.z >= 0.15e-6) & (solver.z <= 2.15 * Lc)
    z_curve = solver.z[mask_curve]
    eta_curve_fit = ((n3 / n1) * np.abs(c3[mask_curve]) ** 2 / np.abs(c1[k_in]) ** 2)
    eta_curve_an = thg_conversion_efficiency(
        I1, z_curve, lam0, chi3, n1=n1, n3=n3, delta_k=dk_disc
    )["efficiency"]
    eta_curve_pm0 = thg_conversion_efficiency(
        I1, z_curve, lam0, chi3, n1=n1, n3=n3, delta_k=0.0
    )["efficiency"]

    k_peak = int(np.argmin(np.abs(solver.z - Lc)))
    k_null = int(np.argmin(np.abs(solver.z - 2.0 * Lc)))
    eta_peak_fit = float((n3 / n1) * np.abs(c3[k_peak]) ** 2 / np.abs(c1[k_in]) ** 2)
    eta_peak_an = float(
        thg_conversion_efficiency(
            I1, float(solver.z[k_peak]), lam0, chi3, n1=n1, n3=n3, delta_k=dk_disc
        )["efficiency"]
    )
    eta_null_fit = float((n3 / n1) * np.abs(c3[k_null]) ** 2 / np.abs(c1[k_in]) ** 2)

    return {
        "f0_hz": f0,
        "n1": n1,
        "n3": n3,
        "delta_k_rad_m": dk_disc,
        "coherence_length_m": Lc,
        "cells_per_lambda_3f0": (c / (3.0 * f0 * n3)) / dz,
        "I1_w_m2": I1,
        "L_over_Lc": (z_curve / Lc).tolist(),
        "eta_fit": eta_curve_fit.tolist(),
        "eta_analytic": np.asarray(eta_curve_an).tolist(),
        "eta_phasematched": np.asarray(eta_curve_pm0).tolist(),
        "eta_peak_fit": eta_peak_fit,
        "eta_peak_analytic": eta_peak_an,
        "peak_rel_error": abs(eta_peak_fit - eta_peak_an) / eta_peak_an,
        "eta_null_fit": eta_null_fit,
        "null_to_peak_ratio": eta_null_fit / eta_peak_fit,
        "wall_time_s": float(time.perf_counter() - t0),
    }


def run_cavity_shift_and_lle_benchmarks():
    """Run 1D/2D Kerr cavity resonance shift and LLE soliton benchmarks."""
    t0 = time.perf_counter()
    L = 1.5e-6
    Nz = 120
    dz = L / Nz
    eps_inf = 2.25
    chi3 = 5.0e-19
    Z_med = float(np.sqrt(mu_0 / (epsilon_0 * eps_inf)))
    grid_1d = wakis.GridFIT3D(
        -0.5e-6, 0.5e-6, -0.5e-6, 0.5e-6, 0.0, L, 2, 2, Nz, verbose=0
    )

    def _run_fp_1d(chi3_val, E_target):
        solver = wakis.SolverFIT3D(
            grid_1d,
            cfln=0.5,
            bc_low=["periodic", "periodic", "pec"],
            bc_high=["periodic", "periodic", "pec"],
            bg=[eps_inf, 1.0, 0.0],
            verbose=0,
        )
        if chi3_val is not None:
            mat = KerrMaterial(eps_inf=eps_inf, chi3=chi3_val, method="newton")
            solver.add_nonlinear_material(mat, center=None, size=None)
        z_e = np.arange(Nz) * dz
        z_h = (np.arange(Nz) + 0.5) * dz
        km = 2.0 * np.pi / L
        sin_km = np.sin(km * z_e)
        cos_km = np.cos(km * z_h)
        H0 = E_target / Z_med
        for i in range(solver.Nx):
            for j in range(solver.Ny):
                solver.H[i, j, :Nz, "y"] = H0 * cos_km
        Nt = 4800
        a_e = np.zeros(Nt)
        e_peak_prof = np.zeros(Nz)
        for n in range(Nt):
            solver.one_step()
            ex_z = solver.E[0, 0, :Nz, "x"]
            a_e[n] = (2.0 / Nz) * np.sum(ex_z * sin_km)
            if abs(a_e[n]) > np.max(e_peak_prof):
                e_peak_prof = np.abs(ex_z).copy()
        stride = 16
        res = wakis.harminv(
            a_e[::stride], solver.dt * stride, fmin=110.0e12, fmax=160.0e12
        )
        idx_dom = int(np.argmax(res.amplitude))
        return float(res.frequency[idx_dom]), e_peak_prof

    E_targets = [2.0e7, 3.5e7, 5.0e7, 6.5e7]
    f_lin_1d, prof_ref = _run_fp_1d(None, E_targets[0])
    V_eff_1d = float(
        effective_nonlinear_volume(
            prof_ref**2, dV=dz * 1.0e-6 * 1.0e-6, eps_r=eps_inf
        )
    )

    df_focus_fit = []
    df_defoc_fit = []
    df_focus_an = []
    df_defoc_an = []
    E_peaks = []

    for E_t in E_targets:
        f_f, _ = _run_fp_1d(chi3, E_t)
        f_d, _ = _run_fp_1d(-chi3, E_t)
        ep = float(np.max(prof_ref)) * (E_t / E_targets[0])
        E_peaks.append(ep)
        df_focus_fit.append(f_f - f_lin_1d)
        df_defoc_fit.append(f_d - f_lin_1d)
        sh_f = kerr_cavity_frequency_shift(
            2.0 * np.pi * f_lin_1d, chi3, E_peak=ep, eps_r=eps_inf, cavity_dim=1
        )["delta_f"]
        df_focus_an.append(float(sh_f))
        df_defoc_an.append(float(-sh_f))

    # LLE single-soliton comb comparison for d2 = 0.01 vs d2 = 0.04
    zeta_0 = 8.0
    f_pump = 3.0
    n_modes = 256
    phi = np.linspace(-np.pi, np.pi, n_modes, endpoint=False)
    lle_results = {}
    for label, d2_val in (("d2_001", 0.01), ("d2_004", 0.04)):
        an = lle_soliton_analytic(phi, zeta_0=zeta_0, d2=d2_val, f_pump=f_pump)
        sim = solve_lle(
            d2=d2_val,
            f_pump=f_pump,
            zeta_0=zeta_0,
            n_modes=n_modes,
            dtau=0.01,
            n_steps=1200,
            psi_init=an["psi"],
            noise_amp=0.0,
        )
        mu = sim["mu"]
        sb_pos = mu > 0
        mu_pos = mu[sb_pos].astype(float)
        p_pos = sim["spectrum"][sb_pos]
        p_norm = p_pos / p_pos[0]
        fwhm_mu_sim = 2.0 * float(np.interp(0.5, p_norm[::-1], mu_pos[::-1]))
        lle_results[label] = {
            "d2": d2_val,
            "mu": mu.tolist(),
            "spectrum_sim": sim["spectrum"].tolist(),
            "spectrum_an": an["spectrum_sech2"].tolist(),
            "fwhm_mu_sim": fwhm_mu_sim,
            "fwhm_mu_an": float(an["fwhm_mu"]),
            "fwhm_rel_error": abs(fwhm_mu_sim - float(an["fwhm_mu"]))
            / float(an["fwhm_mu"]),
        }

    return {
        "f_lin_1d_hz": f_lin_1d,
        "V_eff_1d_m3": V_eff_1d,
        "V_geom_1d_m3": L * 1.0e-6 * 1.0e-6,
        "E_peak_v_m": E_peaks,
        "df_focus_fit_hz": df_focus_fit,
        "df_defoc_fit_hz": df_defoc_fit,
        "df_focus_an_hz": df_focus_an,
        "df_defoc_an_hz": df_defoc_an,
        "cavity_max_rel_error": float(
            np.max(np.abs(np.array(df_focus_fit) - np.array(df_focus_an)) / np.abs(df_focus_an))
        ),
        "lle": lle_results,
        "wall_time_s": float(time.perf_counter() - t0),
    }


def make_figures(data):
    """Generate the two publication-style figures in `docs/img/`."""
    apply_style()
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    c_blue, c_orange, c_green, c_gold, c_purple, _, _, c_black = PALETTE

    # Figure 1: SPM phase shift and THG Maker fringes
    spm = data["spm"]
    thg = data["thg"]
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.4))
    spectrum_grid(ax1)
    spectrum_grid(ax2)

    I_gw_cm2 = np.array(spm["intensity_w_m2"]) * 1e-13  # W/m^2 -> GW/cm^2
    I_dense = np.linspace(0.0, float(np.max(I_gw_cm2)) * 1.05, 100)
    slope_an = spm["dphi_analytic_rad"][-1] / I_gw_cm2[-1]
    slope_wrong = spm["dphi_wrong_factor1_rad"][-1] / I_gw_cm2[-1]

    ax1.plot(
        I_dense,
        slope_an * I_dense * 1e3,
        color=c_blue,
        lw=2.0,
        label=r"Analytic $\Delta\Phi_{\mathrm{NL}} = n_2 k_0 I L$ ($3/4$ factor)",
    )
    ax1.plot(
        I_dense,
        slope_wrong * I_dense * 1e3,
        color=c_black,
        ls="--",
        lw=1.5,
        label=r"Without $3/4$ factor ($\times 4/3$, fails)",
    )
    ax1.plot(
        I_gw_cm2,
        np.array(spm["dphi_newton_rad"]) * 1e3,
        "o",
        color=c_orange,
        ms=6.5,
        label="FIT (Newton cubic solve)",
    )
    ax1.plot(
        [I_gw_cm2[2]],
        [spm["dphi_fixed_point_mid_rad"] * 1e3],
        "s",
        color=c_green,
        ms=6.5,
        fillstyle="none",
        mew=1.8,
        label="FIT (Picard fixed-point)",
    )
    ax1.plot(
        [I_gw_cm2[2]],
        [spm["dphi_explicit_ade_mid_rad"] * 1e3],
        "^",
        color=c_purple,
        ms=6.5,
        fillstyle="none",
        mew=1.8,
        label=r"FIT (Explicit $P_{\mathrm{NL}}$ ADE)",
    )
    ax1.set_xlabel(r"Optical intensity $I$ [$\mathrm{GW/cm^2}$]")
    ax1.set_ylabel(r"Nonlinear phase shift $\Delta\Phi_{\mathrm{NL}}$ [mrad]")
    ax1.set_title("1D Self-Phase Modulation (SPM)")
    ax1.legend(loc="upper left", fontsize=8.5)
    panel_label(ax1, "(a)")

    l_lc = np.array(thg["L_over_Lc"])
    ax2.plot(
        l_lc,
        np.array(thg["eta_analytic"]) * 1e7,
        color=c_blue,
        lw=2.0,
        label=r"Analytic $\propto L^2\,\mathrm{sinc}^2(\Delta k L / 2)$",
    )
    ax2.plot(
        l_lc[::6],
        np.array(thg["eta_fit"])[::6] * 1e7,
        "o",
        color=c_orange,
        ms=5.0,
        label="1D FIT dispersive Kerr slab",
    )
    ax2.axvline(1.0, color=c_green, ls=":", lw=1.4, label=r"$L = L_c = \pi/|\Delta k|$")
    ax2.axvline(2.0, color=c_purple, ls="--", lw=1.3, label=r"Null $L = 2 L_c$")
    ax2.set_xlabel(r"Normalized propagation distance $L / L_c$ [-]")
    ax2.set_ylabel(r"THG efficiency $\eta_{\mathrm{THG}} = I_3 / I_1$ [$10^{-7}$]")
    ax2.set_title(r"Maker-Fringe Third-Harmonic Generation ($3\omega_0$)")
    ax2.legend(loc="upper right", fontsize=8.5)
    panel_label(ax2, "(b)")

    fig1.tight_layout()
    save(fig1, IMG_DIR / "kerr_spm_and_thg.png")
    plt.close(fig1)

    # Figure 2: Kerr cavity resonance shift and LLE soliton microcomb spectrum
    cav = data["cavity_and_lle"]
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(11.2, 4.4))
    spectrum_grid(ax3)
    spectrum_grid(ax4)

    e2_norm = (np.array(cav["E_peak_v_m"]) * 1e-7) ** 2
    e2_dense = np.linspace(0.0, float(np.max(e2_norm)) * 1.05, 100)
    slope_cav = cav["df_focus_an_hz"][-1] * 1e-9 / e2_norm[-1]

    ax3.plot(
        e2_dense,
        slope_cav * e2_dense,
        color=c_blue,
        lw=2.0,
        label=r"Perturbation theory ($\chi^{(3)} > 0$, $V_{\mathrm{eff,NL}} = \frac{2}{3}V$)",
    )
    ax3.plot(
        e2_dense,
        -slope_cav * e2_dense,
        color=c_green,
        ls="--",
        lw=1.8,
        label=r"Perturbation theory ($\chi^{(3)} < 0$, blue-shift)",
    )
    ax3.plot(
        e2_norm,
        np.array(cav["df_focus_fit_hz"]) * 1e-9,
        "o",
        color=c_orange,
        ms=6.5,
        label=r"FIT + harminv ($\chi^{(3)} > 0$)",
    )
    ax3.plot(
        e2_norm,
        np.array(cav["df_defoc_fit_hz"]) * 1e-9,
        "s",
        color=c_purple,
        ms=6.0,
        label=r"FIT + harminv ($\chi^{(3)} < 0$)",
    )
    ax3.axhline(0.0, color=c_black, lw=0.8)
    ax3.set_xlabel(r"Squared modal peak field $E_{\mathrm{peak}}^2$ [$10^{14}\;\mathrm{V^2/m^2}$]")
    ax3.set_ylabel(r"Resonance shift $\Delta f_{\mathrm{NL}} = f_{\mathrm{NL}} - f_0$ [GHz]")
    ax3.set_title("1D Fabry–Pérot Kerr Cavity Frequency Shift")
    ax3.legend(loc="lower left", fontsize=8.5)
    panel_label(ax3, "(a)")

    d2_001 = cav["lle"]["d2_001"]
    d2_004 = cav["lle"]["d2_004"]
    mu = np.array(d2_001["mu"])
    mask_mu = (np.abs(mu) >= 1) & (np.abs(mu) <= 80)

    ax4.plot(
        mu[mask_mu],
        10.0 * np.log10(np.maximum(np.array(d2_001["spectrum_sim"])[mask_mu], 1e-12)),
        ".",
        color=c_blue,
        ms=4.5,
        label=rf"LLE SSFM ($d_2 = 0.01$, $\Delta\mu = {d2_001['fwhm_mu_sim']:.1f}$)",
    )
    ax4.plot(
        mu[mask_mu],
        10.0 * np.log10(np.maximum(np.array(d2_001["spectrum_an"])[mask_mu], 1e-12)),
        color=c_blue,
        lw=1.5,
        alpha=0.8,
        label=r"Analytic $\mathrm{sech}^2$ ($d_2 = 0.01$)",
    )
    ax4.plot(
        mu[mask_mu],
        10.0 * np.log10(np.maximum(np.array(d2_004["spectrum_sim"])[mask_mu], 1e-12)),
        ".",
        color=c_orange,
        ms=4.5,
        label=rf"LLE SSFM ($d_2 = 0.04$, $\Delta\mu = {d2_004['fwhm_mu_sim']:.1f}$)",
    )
    ax4.plot(
        mu[mask_mu],
        10.0 * np.log10(np.maximum(np.array(d2_004["spectrum_an"])[mask_mu], 1e-12)),
        color=c_orange,
        ls="--",
        lw=1.5,
        alpha=0.85,
        label=r"Analytic $\mathrm{sech}^2$ ($d_2 = 0.04$)",
    )
    ax4.set_xlim(-80, 80)
    ax4.set_ylim(-65, -15)
    ax4.set_xlabel(r"Relative azimuthal mode number $\mu$ [-]")
    ax4.set_ylabel(r"Comb line power $|\Psi_\mu|^2$ [dB]")
    ax4.set_title(r"Dissipative Kerr Soliton Comb ($\Delta\mu_{\mathrm{FWHM}} \propto D_2^{-1/2}$)")
    ax4.legend(loc="lower center", fontsize=8.5)
    panel_label(ax4, "(b)")

    fig2.tight_layout()
    save(fig2, IMG_DIR / "kerr_cavity_shift_and_lle.png")
    plt.close(fig2)


def main():
    t_start = time.perf_counter()
    spm = run_spm_benchmark()
    thg = run_thg_maker_fringe_benchmark()
    cav_lle = run_cavity_shift_and_lle_benchmarks()

    payload = {
        "spm": spm,
        "thg": thg,
        "cavity_and_lle": cav_lle,
        "total_wall_time_s": float(time.perf_counter() - t_start),
    }
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    DATA_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    make_figures(payload)
    print(
        f"[013_kerr_nonlinear_optics] SPM max rel err = {spm['max_rel_error']:.3e}, "
        f"THG peak rel err = {thg['peak_rel_error']:.3e}, "
        f"Cavity shift max rel err = {cav_lle['cavity_max_rel_error']:.3e}, "
        f"total time = {payload['total_wall_time_s']:.2f} s"
    )


if __name__ == "__main__":
    main()
