# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Example 014 — Spontaneous-Emission Langevin Noise and Lifetime Acceleration in Gain Media
=========================================================================================

Demonstrates the F11b spontaneous-emission Langevin noise and lifetime-acceleration
(`time_scale`) features of `FourLevelGain` (`wakis.gain`) in `SolverFIT3D` at grid
resolution ``dz = 25 nm`` (``48 cells/lambda`` in vacuum, ``24 cells/lambda`` in
``n_H = 2.0``):

1. **Spontaneous-emission radiated power (1D FDT calibration)**:
   An inverted sub-wavelength gain layer (``L = 50 nm << lambda_0 = 1.2 um``,
   ``dz = 25 nm``) below threshold radiates electromagnetic power matching the
   analytical 1D Fluctuation–Dissipation Theorem (FDT) rate
   ``P_sp = f_rad,1D * N_2 * V * hbar * omega_a / tau_21`` with
   ``f_rad,1D = 3 * lambda_0^2 / (4 * pi * n_bg * A_perp)``, and scales
   quadratically in power with ``noise_amplitude`` (zero power when
   ``noise_amplitude = 0``).
2. **Amplified spontaneous emission (ASE) and gain narrowing**:
   A 1D travelling-wave gain slab (``L_slab = 2.4 um``, ``g_0 * L_slab = 2.80``)
   driven solely by spontaneous-emission Langevin noise reproduces the analytical
   gain-narrowed ASE spectrum ``S(omega) ~ (exp(g(omega)*L) - 1) * g_sp / g`` and
   its narrowed bandwidth compared against the low-gain Lorentzian limit
   (``g_0 * L_slab = 0.15``).
3. **Fabry–Pérot laser self-start from noise, threshold beta-rounding, and mode competition**:
   In a 1D DBR Fabry–Pérot cavity (``dz = 25 nm``, ``24 cells/lambda`` in ``n_H = 2.0``),
   lasing starts from rest without any external seed pulse when
   ``spontaneous_emission=True`` (and produces identically zero field when
   ``spontaneous_emission=False``). Sweeping ``r = W_p / W_{p,th}`` across threshold
   reproduces the rate-equation L–I curve with spontaneous-emission ``beta_sp``
   threshold rounding. In a longer multi-mode cavity (``L_cav = 4.8 um = 8*(lambda_0/2)``),
   multiple longitudinal modes initially grow from spontaneous emission before
   homogeneous gain clamping suppresses side modes by ``> 30 dB`` in steady state.
4. **Lifetime acceleration (`time_scale = s` for s = 1, 10, 100, 1000)**:
   Scaling ``tau_ij -> tau_ij / s`` and ``W_p -> s * W_p`` (plus the stimulated
   transition rate by ``s`` while holding ``kappa_a`` at the physical unscaled
   ``tau_21``) leaves the threshold pump rate, steady-state inversion, saturation
   intensity, steady-state 3D FIT output intensity, and lasing frequency invariant,
   while compressing turn-on delay ``t_th(s) ~ 1/s`` and scaling the Class-B
   relaxation-oscillation frequency as ``f_RO(s) ~ sqrt(s)``.

Saves two publication-style figures in ``docs/img/`` (via `wakis.figstyle`) and
writes all quoted benchmark numbers to ``examples/014_laser_noise_and_time_scaling.json``:
  1. ``docs/img/laser_spontaneous_emission_and_ase.png``
  2. ``docs/img/laser_mode_competition_and_time_scaling.png``
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
from scipy.constants import c as c_light, epsilon_0 as eps_0

import wakis
from wakis import figstyle
from wakis.estimate import estimate, preflight
from wakis.gain import (
    FourLevelGain,
    ase_spectrum_1d,
    dbr_reflectance,
    fabry_perot_laser_output,
    fabry_perot_threshold_gain,
)
from wakis.resonances import ResonanceMonitor
from wakis.sources import CurrentSource, GaussianSource

REPO_ROOT = Path(__file__).resolve().parents[1]
IMG_DIR = REPO_ROOT / "docs" / "img"
JSON_PATH = Path(__file__).resolve().with_suffix(".json")


def make_1d_solver(z_min, z_max, dz, lx=2.0e-6, ly=2.0e-6, cfln=0.8, n_pml=10):
    """Build a 1D-periodic SolverFIT3D along z with grid spacing ``dz`` [m]."""
    nz = int(round((z_max - z_min) / dz))
    est = estimate(
        cells=(2, 2, nz),
        steps=18000,
        dtype="float64",
        backend="cpu",
        cpml=True,
        dispersive_edges=4 * nz,
        n_poles=2,
    )
    preflight(est, "laptop", raise_error=True)

    grid = wakis.GridFIT3D(
        -0.5 * lx,
        0.5 * lx,
        -0.5 * ly,
        0.5 * ly,
        z_min,
        z_max,
        2,
        2,
        nz,
        use_mpi=False,
        verbose=0,
    )
    return wakis.SolverFIT3D(
        grid,
        cfln=cfln,
        bc_low=["periodic", "periodic", "cpml"],
        bc_high=["periodic", "periodic", "cpml"],
        n_pml=n_pml,
        verbose=0,
    )


def add_dbr_mirrors(solver, l_cav, d_h, d_l, n_h, n_pairs):
    """Add symmetric quarter-wave DBR mirrors on both sides of ``[0, l_cav]``.

    With ``n_pairs`` ``(H, L)`` pairs plus ``1`` extra ``H`` layer
    (``extra_high=True``), each mirror has ``2*n_pairs + 1`` quarter-wave layers
    (e.g. ``n_pairs = 1`` builds a 3-layer ``(H L H)`` quarter-wave DBR mirror).
    """
    ieps_arr = solver.ieps.array
    lx_span = 2.0 * float(solver.grid.xmax - solver.grid.xmin)
    ly_span = 2.0 * float(solver.grid.ymax - solver.grid.ymin)
    for m in range(n_pairs + 1):
        z_left = -(m * (d_h + d_l) + 0.5 * d_h)
        z_right = l_cav + m * (d_h + d_l) + 0.5 * d_h
        for z_c in (z_left, z_right):
            idx_e, w_e = solver._box_edges(
                (0.0, 0.0, z_c), (lx_span, ly_span, d_h), dual=False
            )
            eps_s = 1.0 / (eps_0 * ieps_arr[idx_e])
            eps_new = (1.0 - w_e) * eps_s + w_e * (n_h**2)
            ieps_arr[idx_e] = 1.0 / (eps_0 * eps_new)
    solver.update_tensors("ieps")


def run_spontaneous_decay_benchmark():
    """Benchmark 1: 1D inverted thin layer radiated power vs FDT theory."""
    lam0 = 1.2e-6
    f0 = c_light / lam0
    omega0 = 2.0 * np.pi * f0
    gamma_a = 2.5e14
    tau_21_phys = 150.0e-15
    dz = 25.0e-9  # 48 cells/lambda in vacuum
    lx = ly = 2.0e-6
    area_perp = lx * ly
    l_layer = 2.0 * dz  # 50 nm optically thin layer (lambda_0 / 24)
    vol = area_perp * l_layer
    n2_val = 1.0e14
    kappa_a_phys = 6.0 * np.pi * eps_0 * c_light**3 / (omega0**2 * tau_21_phys)

    def measure_thin_layer_power(noise_amp=1.0, n_steps=20000, seed_val=42):
        solver = make_1d_solver(-1.5e-6, 1.5e-6, dz, lx=lx, ly=ly, cfln=0.5)
        mat = FourLevelGain(
            omega_a=omega0,
            gamma_a=gamma_a,
            tau_21=tau_21_phys,
            tau_10=10.0e-15,
            tau_32=5.0e-15,
            kappa_a=kappa_a_phys,
            initial_populations=(1.0e23, 0.0, n2_val, 0.0),
            enable_stimulated=False,
            spontaneous_emission=True,
            noise_amplitude=noise_amp,
            seed=seed_val,
        )
        solver.add_gain_material(
            mat, center=(0.0, 0.0, 0.5 * l_layer), size=(2.0 * lx, 2.0 * ly, l_layer)
        )
        g = solver._gain_groups[0]
        g["A_rate"] = np.eye(4)
        g["B_rate"] = np.zeros((4, 4))

        k_r = int(np.argmin(np.abs(solver.z - (l_layer + 0.4e-6))))
        k_l = int(np.argmin(np.abs(solver.z - (-0.4e-6))))
        stride = solver.Nx * solver.Ny
        ex_r = np.empty(n_steps)
        ex_l = np.empty(n_steps)
        hy_r = np.empty(n_steps)
        hy_l = np.empty(n_steps)
        for n in range(n_steps):
            solver.one_step()
            ex_r[n] = solver.E.array[k_r * stride]
            ex_l[n] = solver.E.array[k_l * stride]
            hy_r[n] = 0.5 * (
                solver.H.array[solver.N + k_r * stride]
                + solver.H.array[solver.N + (k_r - 1) * stride]
            )
            hy_l[n] = 0.5 * (
                solver.H.array[solver.N + k_l * stride]
                + solver.H.array[solver.N + (k_l - 1) * stride]
            )
        n_burn = min(2000, n_steps // 4)
        s_plus = float(
            np.mean(ex_r[n_burn:-1] * 0.5 * (hy_r[n_burn:-1] + hy_r[n_burn + 1 :]))
        )
        s_minus = float(
            np.mean(-ex_l[n_burn:-1] * 0.5 * (hy_l[n_burn:-1] + hy_l[n_burn + 1 :]))
        )
        p_meas = area_perp * (s_plus + s_minus)
        p_theory = mat.spontaneous_emission_power(
            volume=vol, N_2=n2_val, area_perp=area_perp, dt=solver.dt
        )
        return p_meas, p_theory, mat.radiative_fraction_1d(area_perp)

    # Multi-seed ensemble average across 8 independent RNG seeds
    seeds = (101, 102, 103, 104, 105, 106, 107, 108)
    p_seeds = []
    for sd in seeds:
        p_sd, p_th_1, f_rad_1d = measure_thin_layer_power(
            noise_amp=1.0, n_steps=20000, seed_val=sd
        )
        p_seeds.append(float(p_sd))
    p_seeds_arr = np.asarray(p_seeds, dtype=float)
    p_ensemble_mean = float(np.mean(p_seeds_arr))
    sigma_single_rel = float(np.std(p_seeds_arr, ddof=1) / p_th_1)
    sigma_sem_rel = float(sigma_single_rel / np.sqrt(len(seeds)))
    p_seed42, _, _ = measure_thin_layer_power(noise_amp=1.0, n_steps=20000, seed_val=42)

    amps = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    # Consistency check: with fixed seed, linear stochastic response scales as alpha^2 in power
    p_short_1, _, _ = measure_thin_layer_power(noise_amp=1.0, n_steps=5000, seed_val=42)
    p_fit_list = []
    for amp in amps:
        if amp == 0.0:
            p_m, _, _ = measure_thin_layer_power(noise_amp=0.0, n_steps=500, seed_val=42)
            p_fit_list.append(p_m)
        elif np.isclose(amp, 1.0):
            p_fit_list.append(p_ensemble_mean)
        else:
            p_m, _, _ = measure_thin_layer_power(
                noise_amp=float(amp), n_steps=5000, seed_val=42
            )
            p_fit_list.append(p_ensemble_mean * (p_m / p_short_1))

    p_fit_arr = np.array(p_fit_list)
    rel_err_fdt = abs(p_ensemble_mean - p_th_1) / p_th_1
    rel_err_seed42 = abs(p_seed42 - p_th_1) / p_th_1
    rel_err_quad_2 = abs(p_fit_arr[4] / p_ensemble_mean - 4.0) / 4.0
    rel_err_quad_05 = abs(p_fit_arr[1] / p_ensemble_mean - 0.25) / 0.25
    return {
        "amps": amps,
        "p_fit_fW": p_fit_arr * 1e15,
        "p_sp_fit_W": float(p_ensemble_mean),
        "p_sp_single_seed42_W": float(p_seed42),
        "p_sp_seeds_fW": (p_seeds_arr * 1e15).tolist(),
        "n_seeds": len(seeds),
        "sigma_single_seed_rel": sigma_single_rel,
        "sigma_sem_rel": sigma_sem_rel,
        "p_sp_theory_W": float(p_th_1),
        "f_rad_1d": float(f_rad_1d),
        "rel_err_fdt": float(rel_err_fdt),
        "rel_err_seed42": float(rel_err_seed42),
        "rel_err_quad_2": float(rel_err_quad_2),
        "rel_err_quad_05": float(rel_err_quad_05),
    }


def run_ase_spectrum_benchmark():
    """Benchmark 2: 1D gain slab ASE spectrum & gain narrowing vs analytic S_ASE(omega)."""
    lam0 = 1.2e-6
    f0 = c_light / lam0
    omega0 = 2.0 * np.pi * f0
    gamma_a = 3.5e14
    dz = 25.0e-9  # 48 cells/lambda in vacuum
    l_slab = 2.4e-6
    pad = 0.6e-6
    lz = l_slab + 2.0 * pad

    def rms_bandwidth(freqs, psd, f_center=f0, half_band=80.0e12):
        mask = np.abs(freqs - f_center) <= half_band
        f_m = freqs[mask]
        w_m = psd[mask] / np.trapezoid(psd[mask], f_m)
        f_mean = np.trapezoid(f_m * w_m, f_m)
        return float(np.sqrt(np.trapezoid((f_m - f_mean) ** 2 * w_m, f_m)))

    def run_slab_ase(g0_l, nt=18000):
        solver = make_1d_solver(-0.5 * lz, 0.5 * lz, dz)
        mat_ref = FourLevelGain(
            omega_a=omega0,
            gamma_a=gamma_a,
            tau_21=10.0e-12,
            tau_10=20.0e-15,
            tau_32=10.0e-15,
            N_total=1.0e25,
        )
        n2_val = (g0_l / l_slab) / mat_ref.emission_cross_section
        n0_val = 1.0e25 - n2_val
        pops = (n0_val, 0.0, n2_val, 0.0)
        mat = FourLevelGain(
            omega_a=omega0,
            gamma_a=gamma_a,
            tau_21=1.0,
            tau_10=1.0,
            tau_32=1.0,
            kappa_a=mat_ref.kappa_a,
            initial_populations=pops,
            enable_stimulated=False,
            spontaneous_emission=True,
            noise_components="xy",
            seed=100,
        )
        solver.add_gain_material(
            mat, center=(0.0, 0.0, 0.0), size=(4.0e-6, 4.0e-6, l_slab)
        )
        k_r = int(np.argmin(np.abs(solver.z - (0.5 * l_slab + 0.18e-6))))
        k_l = int(np.argmin(np.abs(solver.z - (-0.5 * l_slab - 0.18e-6))))
        idx_r_x = k_r * solver.Nx * solver.Ny
        idx_l_x = k_l * solver.Nx * solver.Ny
        idx_r_y = solver.N + idx_r_x
        idx_l_y = solver.N + idx_l_x

        traces = np.empty((4, nt), dtype=float)
        for n in range(nt):
            solver.one_step()
            traces[0, n] = solver.E.array[idx_r_x]
            traces[1, n] = solver.E.array[idx_l_x]
            traces[2, n] = solver.E.array[idx_r_y]
            traces[3, n] = solver.E.array[idx_l_y]

        nperseg = 2048
        step = 1024
        win = np.hanning(nperseg)
        psd = np.zeros(nperseg // 2 + 1)
        count = 0
        for ch in range(4):
            sig = traces[ch, 1000:]
            for start in range(0, len(sig) - nperseg, step):
                psd += np.abs(np.fft.rfft(sig[start : start + nperseg] * win)) ** 2
                count += 1
        psd /= count
        freqs = np.fft.rfftfreq(nperseg, d=solver.dt)
        ase_th = ase_spectrum_1d(freqs, l_slab, mat, populations=pops, dt=solver.dt)
        return freqs, psd, ase_th, mat, pops

    freqs_lo, psd_lo, ase_lo, mat_lo, pops_lo = run_slab_ase(0.15)
    freqs_hi, psd_hi, ase_hi, mat_hi, pops_hi = run_slab_ase(2.80)

    w_fit_lo = rms_bandwidth(freqs_lo, psd_lo) * 1e-12
    w_th_lo = rms_bandwidth(freqs_lo, ase_lo) * 1e-12
    w_fit_hi = rms_bandwidth(freqs_hi, psd_hi) * 1e-12
    w_th_hi = rms_bandwidth(freqs_hi, ase_hi) * 1e-12
    # ase_linewidth already returns Hz (gamma_a / (2 pi) * sqrt(G / ln((e^G + 1)/2) - 1))
    fwhm_an_hi_THz = float(mat_hi.ase_linewidth(l_slab, populations=pops_hi) * 1e-12)
    fwhm_ideal_lorentz_THz = float(
        (gamma_a / (2.0 * np.pi))
        * np.sqrt(2.80 / np.log(0.5 * (np.exp(2.80) + 1.0)) - 1.0)
        * 1e-12
    )
    fwhm_hom_THz = float(gamma_a / (2.0 * np.pi) * 1e-12)

    band = (freqs_hi >= 170.0e12) & (freqs_hi <= 330.0e12)
    f_band_THz = freqs_hi[band] * 1e-12
    s_fit_hi = psd_hi[band] / np.max(psd_hi[band])
    s_an_hi = ase_hi[band] / np.max(ase_hi[band])
    s_an_lo = ase_lo[band] / np.max(ase_lo[band])
    rel_err_rms_hi = abs(w_fit_hi - w_th_hi) / w_th_hi
    rel_err_rms_lo = abs(w_fit_lo - w_th_lo) / w_th_lo

    return {
        "f_THz": f_band_THz,
        "s_fit": s_fit_hi,
        "s_an": s_an_hi,
        "s_lorentz": s_an_lo,
        "g0_L": 2.80,
        "rms_fit_hi_THz": w_fit_hi,
        "rms_th_hi_THz": w_th_hi,
        "rms_fit_lo_THz": w_fit_lo,
        "rms_th_lo_THz": w_th_lo,
        "fwhm_an_THz": fwhm_an_hi_THz,
        "fwhm_ideal_lorentz_THz": fwhm_ideal_lorentz_THz,
        "fwhm_hom_THz": fwhm_hom_THz,
        "rel_err_rms_hi": rel_err_rms_hi,
        "rel_err_rms_lo": rel_err_rms_lo,
    }


def run_fp_noise_and_mode_competition():
    """Benchmark 3 & 4: FP lasing from noise + beta_sp L-I curve + multi-mode competition."""
    lam0 = 1.2e-6
    f0 = c_light / lam0
    omega0 = 2.0 * np.pi * f0
    gamma_a = 1.2e14
    tau_21 = 150.0e-15
    tau_10 = 5.0e-15
    tau_32 = 5.0e-15
    n_total = 5.0e23
    dz = 25.0e-9  # 24 cells/lambda in n_H = 2.0 (>= 20 cells/lambda rule)
    n_h, n_l = 2.0, 1.0
    d_h = lam0 / (4.0 * n_h)
    d_l = lam0 / (4.0 * n_l)
    n_pairs = 1
    l_cav = 1.2e-6

    r_dbr = dbr_reflectance(n_h, n_l, n_pairs, n_in=1.0, n_out=1.0, extra_high=True)
    g_th = fabry_perot_threshold_gain(l_cav, r_dbr, r_dbr)

    fg_ref = FourLevelGain(
        omega_a=omega0,
        gamma_a=gamma_a,
        tau_21=tau_21,
        tau_10=tau_10,
        tau_32=tau_32,
        N_total=n_total,
    )

    lx_fp = ly_fp = 3.6e-6
    area_perp_fp = lx_fp * ly_fp
    # A priori theoretical spontaneous-emission coupling factor beta_sp,theory
    # from 1D radiative fraction f_rad,1D(A_perp), longitudinal confinement
    # Gamma_z = L_cav / L_eff (with DBR penetration L_pen = lambda_0 / (4*(n_H - n_L))),
    # and Lorentzian spectral overlap (2/pi)*arctan(Delta omega_FSR / gamma_a):
    l_pen_fp = lam0 / (4.0 * (n_h - n_l))
    l_eff_fp = l_cav + 2.0 * l_pen_fp
    gamma_z_fp = l_cav / l_eff_fp
    dw_fsr_fp = np.pi * c_light / l_eff_fp
    spectral_overlap_fp = float((2.0 / np.pi) * np.arctan(dw_fsr_fp / gamma_a))
    f_rad_fp = float(fg_ref.radiative_fraction_1d(area_perp_fp))
    beta_theory = float(f_rad_fp * gamma_z_fp * spectral_overlap_fp)

    def run_fp_noise_point(
        pump_ratio, spontaneous=True, total_time=1.25e-12, noise_amp=1.0
    ):
        wp = fg_ref.threshold_pump_rate(pump_ratio * g_th)
        solver = make_1d_solver(-1.2e-6, 2.4e-6, dz, lx=lx_fp, ly=ly_fp)
        add_dbr_mirrors(solver, l_cav, d_h, d_l, n_h, n_pairs)
        fg = FourLevelGain(
            omega_a=omega0,
            gamma_a=gamma_a,
            tau_21=tau_21,
            tau_10=tau_10,
            tau_32=tau_32,
            N_total=n_total,
            pump_rate=wp,
            initial_populations="steady_state",
            spatial_averaging=True,
            spontaneous_emission=spontaneous,
            noise_amplitude=noise_amp if spontaneous else 0.0,
            seed=42,
        )
        solver.add_gain_material(
            fg,
            center=(0.0, 0.0, 0.5 * l_cav),
            size=(2.0 * lx_fp, 2.0 * ly_fp, l_cav),
        )
        z_out = l_cav + 2.0 * d_h + d_l + 0.2e-6
        k_out = int(np.argmin(np.abs(solver.z - z_out)))
        idx_out = k_out * solver.Nx * solver.Ny
        nt = int(total_time / solver.dt)
        ex_out = np.empty(nt)
        for n in range(nt):
            solver.one_step()
            ex_out[n] = solver.E.array[idx_out]
        i_out = float(c_light * eps_0 * np.mean(ex_out[int(0.7 * nt) :] ** 2))
        return i_out, fg.saturation_intensity(W_p=wp)

    i_zero_noise, _ = run_fp_noise_point(2.5, spontaneous=False, total_time=0.20e-12)

    r_sweep = np.array([0.5, 0.85, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0])
    i_meas = np.empty_like(r_sweep)
    i_sats = np.empty_like(r_sweep)
    for i, r_val in enumerate(r_sweep):
        i_meas[i], i_sats[i] = run_fp_noise_point(float(r_val), spontaneous=True)

    i_scale = 0.5 * np.log(1.0 / r_dbr) * i_sats
    s_meas = i_meas / i_scale
    beta_eff = float(s_meas[r_sweep == 1.0][0] ** 2)

    i_theory_beta = np.array(
        [
            fabry_perot_laser_output(
                r, isat, r_dbr, r_dbr, spatial_hole_burning=False, beta_sp=beta_eff
            )
            for r, isat in zip(r_sweep, i_sats)
        ]
    )
    i_theory_apriori = np.array(
        [
            fabry_perot_laser_output(
                r, isat, r_dbr, r_dbr, spatial_hole_burning=False, beta_sp=beta_theory
            )
            for r, isat in zip(r_sweep, i_sats)
        ]
    )
    i_theory_sharp = np.array(
        [
            fabry_perot_laser_output(
                r, isat, r_dbr, r_dbr, spatial_hole_burning=False, beta_sp=0.0
            )
            for r, isat in zip(r_sweep, i_sats)
        ]
    )
    l2_err_beta = float(
        np.linalg.norm(i_meas - i_theory_beta) / np.linalg.norm(i_theory_beta)
    )
    oos_mask = r_sweep != 1.0
    l2_err_beta_oos = float(
        np.linalg.norm(i_meas[oos_mask] - i_theory_beta[oos_mask])
        / np.linalg.norm(i_theory_beta[oos_mask])
    )
    l2_err_beta_apriori = float(
        np.linalg.norm(i_meas - i_theory_apriori) / np.linalg.norm(i_theory_apriori)
    )
    l2_err_beta_sharp = float(
        np.linalg.norm(i_meas - i_theory_sharp) / np.linalg.norm(i_theory_sharp)
    )
    # Sub-threshold and threshold points (r <= 1.0) are the only regime sensitive
    # to beta_sp; above-threshold points (r >= 1.5) dominate the full-range L2 norm.
    sub_th_mask = r_sweep <= 1.0
    l2_err_sub_th_apriori = float(
        np.linalg.norm(i_meas[sub_th_mask] - i_theory_apriori[sub_th_mask])
        / np.linalg.norm(i_theory_apriori[sub_th_mask])
    )
    l2_err_sub_th_eff = float(
        np.linalg.norm(i_meas[sub_th_mask] - i_theory_beta[sub_th_mask])
        / np.linalg.norm(i_theory_beta[sub_th_mask])
    )
    l2_err_sub_th_sharp = float(
        np.linalg.norm(i_meas[sub_th_mask] - i_theory_sharp[sub_th_mask])
        / np.linalg.norm(i_meas[sub_th_mask])
    )
    idx_r1 = int(np.where(r_sweep == 1.0)[0][0])
    idx_r085 = int(np.where(r_sweep == 0.85)[0][0])
    rel_err_r1_apriori = float(
        abs(i_meas[idx_r1] - i_theory_apriori[idx_r1]) / i_theory_apriori[idx_r1]
    )
    rel_err_r085_apriori = float(
        abs(i_meas[idx_r085] - i_theory_apriori[idx_r085]) / i_theory_apriori[idx_r085]
    )
    rel_diff_beta_apriori_vs_eff = float(abs(beta_theory - beta_eff) / beta_eff)

    r_dense = np.linspace(0.05, 3.1, 140)
    i_an_beta_dense = np.array(
        [
            fabry_perot_laser_output(
                r,
                fg_ref.saturation_intensity(W_p=fg_ref.threshold_pump_rate(r * g_th)),
                r_dbr,
                r_dbr,
                spatial_hole_burning=False,
                beta_sp=beta_eff,
            )
            for r in r_dense
        ]
    )
    i_an_apriori_dense = np.array(
        [
            fabry_perot_laser_output(
                r,
                fg_ref.saturation_intensity(W_p=fg_ref.threshold_pump_rate(r * g_th)),
                r_dbr,
                r_dbr,
                spatial_hole_burning=False,
                beta_sp=beta_theory,
            )
            for r in r_dense
        ]
    )
    i_an_sharp_dense = np.array(
        [
            fabry_perot_laser_output(
                r,
                fg_ref.saturation_intensity(W_p=fg_ref.threshold_pump_rate(r * g_th)),
                r_dbr,
                r_dbr,
                spatial_hole_burning=False,
                beta_sp=0.0,
            )
            for r in r_dense
        ]
    )

    # Multi-mode cavity mode competition (L_mm = 4.8 um = 4 * lambda_0)
    l_mm = 4.8e-6
    gamma_mm = 2.8e14
    g_th_mm = fabry_perot_threshold_gain(l_mm, r_dbr, r_dbr)
    fg_mm_ref = FourLevelGain(
        omega_a=omega0,
        gamma_a=gamma_mm,
        tau_21=tau_21,
        tau_10=tau_10,
        tau_32=tau_32,
        N_total=n_total,
    )
    wp_mm = fg_mm_ref.threshold_pump_rate(2.5 * g_th_mm)
    solver_mm = make_1d_solver(-1.2e-6, l_mm + 1.2e-6, dz)
    add_dbr_mirrors(solver_mm, l_mm, d_h, d_l, n_h, n_pairs)
    fg_mm = FourLevelGain(
        omega_a=omega0,
        gamma_a=gamma_mm,
        tau_21=tau_21,
        tau_10=tau_10,
        tau_32=tau_32,
        N_total=n_total,
        pump_rate=wp_mm,
        initial_populations="steady_state",
        spatial_averaging=True,
        spontaneous_emission=True,
        noise_amplitude=0.05,
        seed=7,
    )
    solver_mm.add_gain_material(
        fg_mm, center=(0.0, 0.0, 0.5 * l_mm), size=(4.0e-6, 4.0e-6, l_mm)
    )
    z_out_mm = l_mm + 2.0 * d_h + d_l + 0.2e-6
    k_out_mm = int(np.argmin(np.abs(solver_mm.z - z_out_mm)))
    idx_out_mm = k_out_mm * solver_mm.Nx * solver_mm.Ny
    nt_mm = int(2.4e-12 / solver_mm.dt)
    ex_mm = np.empty(nt_mm)
    for n in range(nt_mm):
        solver_mm.one_step()
        ex_mm[n] = solver_mm.E.array[idx_out_mm]

    dt_mm = solver_mm.dt
    nfft = 32768
    freqs = np.fft.rfftfreq(nfft, d=dt_mm)
    band_mm = (freqs >= 195.0e12) & (freqs <= 305.0e12)
    f_mm_THz = freqs[band_mm] * 1e-12

    seg_early = ex_mm[int(0.05e-12 / dt_mm) : int(0.45e-12 / dt_mm)]
    seg_late = ex_mm[int(1.5e-12 / dt_mm) :]
    spec_early = (
        np.abs(np.fft.rfft(seg_early * np.hanning(len(seg_early)), n=nfft))[band_mm] ** 2
    )
    spec_late = (
        np.abs(np.fft.rfft(seg_late * np.hanning(len(seg_late)), n=nfft))[band_mm] ** 2
    )
    spec_early /= np.max(spec_early)
    spec_late /= np.max(spec_late)

    f_peak_mm = float(f_mm_THz[np.argmax(spec_late)])
    main_lobe = np.abs(f_mm_THz - f_peak_mm) <= 8.0
    dominant_mode_fraction = float(np.sum(spec_late[main_lobe]) / np.sum(spec_late))
    m_side = ((f_mm_THz >= 212.0) & (f_mm_THz <= 232.0)) | (
        (f_mm_THz >= 268.0) & (f_mm_THz <= 288.0)
    )
    side_early = float(np.max(spec_early[m_side]))
    side_late = float(np.max(spec_late[m_side]))
    smsr_dB = float(10.0 * np.log10(1.0 / max(side_late, 1e-12)))

    return {
        "r_sweep": r_sweep,
        "i_fit_MW_m2": i_meas * 1e-6,
        "r_dense": r_dense,
        "i_an_beta_MW_m2": i_an_beta_dense * 1e-6,
        "i_an_apriori_MW_m2": i_an_apriori_dense * 1e-6,
        "i_an_sharp_MW_m2": i_an_sharp_dense * 1e-6,
        "beta_eff": beta_eff,
        "beta_theory": beta_theory,
        "rel_diff_beta_apriori_vs_eff": rel_diff_beta_apriori_vs_eff,
        "f_rad_fp": f_rad_fp,
        "gamma_z_fp": gamma_z_fp,
        "spectral_overlap_fp": spectral_overlap_fp,
        "i_zero_noise": i_zero_noise,
        "l2_err_beta": l2_err_beta,
        "l2_err_beta_oos": l2_err_beta_oos,
        "l2_err_beta_apriori": l2_err_beta_apriori,
        "l2_err_beta_sharp": l2_err_beta_sharp,
        "l2_err_sub_th_apriori": l2_err_sub_th_apriori,
        "l2_err_sub_th_eff": l2_err_sub_th_eff,
        "l2_err_sub_th_sharp": l2_err_sub_th_sharp,
        "rel_err_r1_apriori": rel_err_r1_apriori,
        "rel_err_r085_apriori": rel_err_r085_apriori,
        "f_mm_THz": f_mm_THz,
        "spec_early": spec_early,
        "spec_late": spec_late,
        "f_peak_mm_THz": f_peak_mm,
        "dominant_mode_fraction": dominant_mode_fraction,
        "side_early": side_early,
        "side_late": side_late,
        "smsr_dB": smsr_dB,
    }


def run_lifetime_acceleration_benchmark():
    """Benchmark 5: Lifetime acceleration (`time_scale = s` for s = 1, 10, 100, 1000)."""
    lam0 = 1.2e-6
    f0 = c_light / lam0
    omega0 = 2.0 * np.pi * f0
    gamma_a = 1.0e14
    dz = 25.0e-9  # 24 cells/lambda in n_H = 2.0
    n_h, n_l = 2.0, 1.0
    d_h = lam0 / (4.0 * n_h)
    d_l = lam0 / (4.0 * n_l)
    n_pairs = 1
    l_cav = 1.2e-6

    r_dbr = dbr_reflectance(n_h, n_l, n_pairs, n_in=1.0, n_out=1.0, extra_high=True)
    g_th = fabry_perot_threshold_gain(l_cav, r_dbr, r_dbr)

    # Cold-cavity ringdown (gain off, ResonanceMonitor / harminv) on the same
    # passive 3-layer (H L H) DBR cavity to measure kappa_cav and tau_p_cold:
    sol_ring = make_1d_solver(-1.2e-6, 2.4e-6, dz, cfln=0.8, n_pml=10)
    add_dbr_mirrors(sol_ring, l_cav, d_h, d_l, n_h, n_pairs)
    src_ring = GaussianSource(frequency=f0, fwidth=0.05 * f0)
    sheet_ring = CurrentSource(
        src_ring,
        component="x",
        center=(0.0, 0.0, 0.25 * l_cav),
        size=(2.0e-6, 2.0e-6, 0.0),
        amplitude=1.0,
    )
    mon_ring = ResonanceMonitor(
        component="Ex",
        center=(0.0, 0.0, 0.75 * l_cav),
        t_start=src_ring.end_time + 30.0e-15,
    )
    nt_ring = int((src_ring.end_time + 300.0e-15) / sol_ring.dt)
    sol_ring.emsolve(nt_ring, source=sheet_ring, monitors=[mon_ring])
    modes_ring = mon_ring.harminv(0.85 * f0, 1.15 * f0)
    idx_cold = int(np.argmin(np.abs(modes_ring.frequency - f0)))
    f_cold_val = float(modes_ring.frequency[idx_cold])
    q_cold_val = float(modes_ring.Q[idx_cold])
    kappa_cav = float(2.0 * modes_ring.decay_rate[idx_cold])
    tau_p_cold = 1.0 / kappa_cav

    l_pen = lam0 / (4.0 * (n_h - n_l))
    gamma_z = l_cav / (l_cav + 2.0 * l_pen)
    # Active-region Lorentzian polarization slow-light group delay enhancement:
    tau_p_disp = tau_p_cold * (1.0 + gamma_z * kappa_cav / gamma_a)

    tau_21 = 15.0e-12
    tau_10 = 20.0e-15
    tau_32 = 10.0e-15
    n_total = 6.0e25
    r_pump = 3.0

    ref_mat = FourLevelGain(
        omega_a=omega0,
        gamma_a=gamma_a,
        tau_21=tau_21,
        tau_10=tau_10,
        tau_32=tau_32,
        N_total=n_total,
    )
    wp_op = ref_mat.threshold_pump_rate(r_pump * g_th)
    ref_mat.W_p = wp_op
    isat_ref = ref_mat.saturation_intensity()

    scales = (1.0, 10.0, 100.0, 1000.0)
    turn_on_traces = {}
    t_cross_num = []
    t_th_theory = []
    f_ro_theory_THz = []
    f_ro_undamped_disp_THz = []
    f_ro_damped_bare_THz = []
    f_ro_damped_disp_THz = []
    t_quarter_bare_fs = []
    t_quarter_disp_fs = []
    t_quarter_undamped_disp_fs = []
    limits_by_s = {}

    for s in scales:
        mat_s = FourLevelGain(
            omega_a=omega0,
            gamma_a=gamma_a,
            tau_21=tau_21,
            tau_10=tau_10,
            tau_32=tau_32,
            N_total=n_total,
            pump_rate=wp_op,
            time_scale=s,
        )
        t_th_s = mat_s.turn_on_delay(g_th)
        t_th_theory.append(float(t_th_s))
        f_ro_theory_THz.append(
            float(
                mat_s.relaxation_oscillation_frequency(r_pump, tau_p_cold, damped=False)
                * 1e-12
            )
        )
        f_d_undamped = float(
            mat_s.relaxation_oscillation_frequency(r_pump, tau_p_disp, damped=False)
        )
        f_ro_undamped_disp_THz.append(f_d_undamped * 1e-12)
        t_quarter_undamped_disp_fs.append(float(0.25 / f_d_undamped * 1e15))
        f_b = float(
            mat_s.relaxation_oscillation_frequency(r_pump, tau_p_cold, damped=True)
        )
        f_d = float(
            mat_s.relaxation_oscillation_frequency(r_pump, tau_p_disp, damped=True)
        )
        f_ro_damped_bare_THz.append(f_b * 1e-12)
        f_ro_damped_disp_THz.append(f_d * 1e-12)
        t_quarter_bare_fs.append(float(0.25 / f_b * 1e15) if f_b > 0.0 else 0.0)
        t_quarter_disp_fs.append(float(0.25 / f_d * 1e15) if f_d > 0.0 else 0.0)
        limits_by_s[str(int(s))] = mat_s.time_scaling_limits(tau_p=tau_p_cold)

        dt_rate = 0.2e-15
        nt_rate = int(round(2.4 * t_th_s / dt_rate))
        t_arr, n_hist = mat_s.step_rate_equations(nt_rate, dt_rate)
        dn_arr = n_hist[:, 2] - n_hist[:, 1]
        dn_th = g_th / mat_s.emission_cross_section
        idx_c = int(np.argmax(dn_arr >= dn_th))
        t_c = t_arr[idx_c - 1] + (
            (dn_th - dn_arr[idx_c - 1]) / (dn_arr[idx_c] - dn_arr[idx_c - 1])
        ) * dt_rate
        t_cross_num.append(float(t_c))
        turn_on_traces[s] = (t_arr * s * 1e12, dn_arr / dn_th)

    # Independent cold-start 3D FIT steady-state convergence check across s = 1, 5, 20
    tau_21_cold = 150.0e-15
    ref_cold = FourLevelGain(
        omega_a=omega0,
        gamma_a=gamma_a,
        tau_21=tau_21_cold,
        tau_10=5.0e-15,
        tau_32=5.0e-15,
        N_total=8.0e23,
    )
    wp_cold = ref_cold.threshold_pump_rate(2.5 * g_th)
    isat_cold = ref_cold.saturation_intensity(W_p=wp_cold)
    i_out_cold_th = float(
        fabry_perot_laser_output(
            2.5, isat_cold, r_dbr, r_dbr, spatial_hole_burning=False
        )
    )
    cold_scales = (1.0, 5.0, 20.0)
    cold_i_outs_MW_m2 = []
    cold_dn_ss_m3 = []
    for sv in cold_scales:
        sol_c = make_1d_solver(-1.2e-6, 2.4e-6, dz, cfln=0.8, n_pml=10)
        add_dbr_mirrors(sol_c, l_cav, d_h, d_l, n_h, n_pairs)
        m_c = FourLevelGain(
            omega_a=omega0,
            gamma_a=gamma_a,
            tau_21=tau_21_cold,
            tau_10=5.0e-15,
            tau_32=5.0e-15,
            N_total=8.0e23,
            pump_rate=wp_cold,
            initial_populations="steady_state",
            spatial_averaging=True,
            time_scale=sv,
        )
        sol_c.add_gain_material(
            m_c, center=(0.0, 0.0, 0.5 * l_cav), size=(4.0e-6, 4.0e-6, l_cav)
        )
        src_c = GaussianSource(frequency=f0, fwidth=0.08 * f0)
        sheet_c = CurrentSource(
            src_c,
            component="x",
            center=(0.0, 0.0, 0.25 * l_cav),
            size=(2.0e-6, 2.0e-6, 0.0),
            amplitude=2.0,
        )
        z_out_c = l_cav + 2.0 * d_h + d_l + 0.2e-6
        k_out_c = int(np.argmin(np.abs(sol_c.z - z_out_c)))
        idx_out_c = k_out_c * sol_c.Nx * sol_c.Ny
        g_c = sol_c._gain_groups[0]
        cw_c = g_c["cell_weight"] / g_c["eff_n_cells"]
        nt_c = int(1.4e-12 / sol_c.dt)
        nt_pulse = int(0.15e-12 / sol_c.dt)
        ex_c = np.empty(nt_c)
        dn_c = np.empty(nt_c)
        for n in range(nt_c):
            if n < nt_pulse:
                sheet_c.update(sol_c, n * sol_c.dt)
            elif n == nt_pulse:
                sol_c.J.array[:] = 0.0
            sol_c.one_step()
            ex_c[n] = sol_c.E.array[idx_out_c]
            dn_c[n] = float((g_c["N"][2] - g_c["N"][1]) @ cw_c)
        n_tail = int(0.4e-12 / sol_c.dt)
        cold_i_outs_MW_m2.append(
            float(c_light * eps_0 * np.mean(ex_c[-n_tail:] ** 2) * 1e-6)
        )
        cold_dn_ss_m3.append(float(np.mean(dn_c[-n_tail:])))

    solver_warm = make_1d_solver(-1.2e-6, 2.4e-6, dz, cfln=0.8, n_pml=10)
    add_dbr_mirrors(solver_warm, l_cav, d_h, d_l, n_h, n_pairs)
    mat_warm = FourLevelGain(
        omega_a=omega0,
        gamma_a=gamma_a,
        tau_21=tau_21,
        tau_10=tau_10,
        tau_32=tau_32,
        N_total=n_total,
        pump_rate=wp_op,
        initial_populations="steady_state",
        spatial_averaging=True,
        time_scale=200.0,
    )
    solver_warm.add_gain_material(
        mat_warm, center=(0.0, 0.0, 0.5 * l_cav), size=(4.0e-6, 4.0e-6, l_cav)
    )
    src = GaussianSource(frequency=f0, fwidth=0.08 * f0)
    sheet = CurrentSource(
        src,
        component="x",
        center=(0.0, 0.0, 0.25 * l_cav),
        size=(2.0e-6, 2.0e-6, 0.0),
        amplitude=2.0,
    )
    nt_warm = int(0.8e-12 / solver_warm.dt)
    for n in range(nt_warm):
        sheet.update(solver_warm, n * solver_warm.dt)
        solver_warm.one_step()
    solver_warm.J.array[:] = 0.0
    for _ in range(int(0.2e-12 / solver_warm.dt)):
        solver_warm.one_step()

    g_warm = solver_warm._gain_groups[0]
    e_ss = solver_warm.E.array.copy()
    h_ss = solver_warm.H.array.copy()
    psi_ss = {
        a: getattr(solver_warm, a).copy()
        for a in dir(solver_warm)
        if a.startswith("psi")
    }
    pa_ss = g_warm["lorentz_a"][3].copy()
    pa_old_ss = g_warm["lorentz_a"][4].copy()
    n_ss_fit = g_warm["N"].copy()
    dn_ss_val = float(np.mean(n_ss_fit[2] - n_ss_fit[1]))
    delta_n = 0.05 * dn_ss_val
    n_pert = n_ss_fit.copy()
    n_pert[2] += delta_n
    n_pert[0] -= delta_n

    i_out_theory = float(
        fabry_perot_laser_output(
            r_pump, isat_ref, r_dbr, r_dbr, spatial_hole_burning=False
        )
    )
    z_out = l_cav + 2.0 * d_h + d_l + 0.2e-6
    k_out = int(np.argmin(np.abs(solver_warm.z - z_out)))
    idx_out = k_out * solver_warm.Nx * solver_warm.Ny
    cw = g_warm["cell_weight"] / g_warm["eff_n_cells"]
    dt = solver_warm.dt
    n_avg = int((20.0 / f0) / dt)

    def restore_state(n_init_arr, s_val):
        solver_warm.E.array[:] = e_ss
        solver_warm.H.array[:] = h_ss
        for a, v in psi_ss.items():
            getattr(solver_warm, a)[:] = v
        g_warm["lorentz_a"][3][:] = pa_ss
        g_warm["lorentz_a"][4][:] = pa_old_ss
        g_warm["N"][:] = n_init_arr
        mat_warm.time_scale = s_val

    i_outs_MW_m2 = []
    dn_ss_list = []
    f_peaks_THz = []
    t_quarter_fs = []
    sim_windows = {1.0: 1.5e-12, 10.0: 0.75e-12, 100.0: 0.40e-12, 1000.0: 0.30e-12}

    for s in scales:
        restore_state(n_ss_fit, s)
        ex_ss = np.empty(2 * n_avg)
        dn_ss_h = np.empty(2 * n_avg)
        for n in range(2 * n_avg):
            solver_warm.one_step()
            ex_ss[n] = solver_warm.E.array[idx_out]
            dn_ss_h[n] = float((g_warm["N"][2] - g_warm["N"][1]) @ cw)
        i_outs_MW_m2.append(
            float(c_light * eps_0 * np.mean(ex_ss[-n_avg:] ** 2) * 1e-6)
        )
        dn_ss_list.append(float(np.mean(dn_ss_h[-n_avg:])))
        spec = np.abs(np.fft.rfft(ex_ss * np.hanning(2 * n_avg), n=32768))
        freqs = np.fft.rfftfreq(32768, d=dt)
        f_peaks_THz.append(float(freqs[np.argmax(spec)] * 1e-12))

        restore_state(n_pert, s)
        nt = int(sim_windows[s] / dt)
        dn_trace = np.empty(nt)
        for n in range(nt):
            solver_warm.one_step()
            dn_trace[n] = float((g_warm["N"][2] - g_warm["N"][1]) @ cw) - dn_ss_val
        win = int(round(2.0 / (f0 * dt)))
        dn_smooth = np.convolve(dn_trace, np.ones(win) / win, mode="valid")
        t_smooth = np.arange(len(dn_smooth)) * dt
        zc = np.where(np.diff(np.sign(dn_smooth)) != 0)[0]
        t_quarter_fs.append(float(t_smooth[zc[0]] * 1e15))

    t_th_rel_err = [
        float(abs(tn - tt) / tt) for tn, tt in zip(t_cross_num, t_th_theory)
    ]
    rel_err_ro_vs_disp_damped = [
        float((t_fit - t_th) / t_th)
        for t_fit, t_th in zip(t_quarter_fs[:3], t_quarter_disp_fs[:3])
    ]
    rel_err_ro_vs_disp_undamped = [
        float((t_fit - t_th) / t_th)
        for t_fit, t_th in zip(t_quarter_fs[:3], t_quarter_undamped_disp_fs[:3])
    ]
    ratio_t_quarter_1_to_10_fit = float(t_quarter_fs[0] / t_quarter_fs[1])
    ratio_t_quarter_1_to_100_fit = float(t_quarter_fs[0] / t_quarter_fs[2])
    ratio_t_quarter_1_to_10_disp_damped = float(
        t_quarter_disp_fs[0] / t_quarter_disp_fs[1]
    )
    ratio_t_quarter_1_to_100_disp_damped = float(
        t_quarter_disp_fs[0] / t_quarter_disp_fs[2]
    )

    return {
        "scales": list(scales),
        "f_cold_THz": float(f_cold_val * 1e-12),
        "Q_cold": float(q_cold_val),
        "kappa_cav_s_inv": float(kappa_cav),
        "tau_p_cold_fs": float(tau_p_cold * 1e15),
        "tau_p_disp_fs": float(tau_p_disp * 1e15),
        "t_th_theory_ps": [v * 1e12 for v in t_th_theory],
        "t_th_num_ps": [v * 1e12 for v in t_cross_num],
        "t_th_rel_err": t_th_rel_err,
        "f_ro_theory_THz": f_ro_theory_THz,
        "f_ro_undamped_disp_THz": f_ro_undamped_disp_THz,
        "f_ro_damped_bare_THz": f_ro_damped_bare_THz,
        "f_ro_damped_disp_THz": f_ro_damped_disp_THz,
        "t_quarter_bare_fs": t_quarter_bare_fs,
        "t_quarter_disp_fs": t_quarter_disp_fs,
        "t_quarter_undamped_disp_fs": t_quarter_undamped_disp_fs,
        "rel_err_ro_vs_disp_damped": rel_err_ro_vs_disp_damped,
        "rel_err_ro_vs_disp_undamped": rel_err_ro_vs_disp_undamped,
        "ratio_t_quarter_1_to_10_fit": ratio_t_quarter_1_to_10_fit,
        "ratio_t_quarter_1_to_100_fit": ratio_t_quarter_1_to_100_fit,
        "ratio_t_quarter_1_to_10_disp_damped": ratio_t_quarter_1_to_10_disp_damped,
        "ratio_t_quarter_1_to_100_disp_damped": ratio_t_quarter_1_to_100_disp_damped,
        "i_outs_MW_m2": i_outs_MW_m2,
        "i_out_theory_MW_m2": i_out_theory * 1e-6,
        "dn_ss_list": dn_ss_list,
        "f_peaks_THz": f_peaks_THz,
        "t_quarter_fs": t_quarter_fs,
        "turn_on_traces": turn_on_traces,
        "limits_by_s": limits_by_s,
        "cold_start_scales": list(cold_scales),
        "cold_start_i_outs_MW_m2": cold_i_outs_MW_m2,
        "cold_start_i_out_theory_MW_m2": i_out_cold_th * 1e-6,
        "cold_start_dn_ss_m3": cold_dn_ss_m3,
        "r_pump": r_pump,
        "tau_p_disp": tau_p_disp,
        "ref_mat_params": {
            "omega_a": omega0,
            "gamma_a": gamma_a,
            "tau_21": tau_21,
            "tau_10": tau_10,
            "tau_32": tau_32,
            "N_total": n_total,
            "pump_rate": wp_op,
        },
    }


def make_figures(b1, b2, b34, b5):
    """Create and save the two 3-panel publication figures via `wakis.figstyle`."""
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Figure 1: Spontaneous-emission FDT power, 1D ASE gain narrowing,
    #           and FP L-I curve with spontaneous-emission beta rounding
    # ------------------------------------------------------------------
    fig1, axes1 = figstyle.new_figure(width_in=14.0, height_in=4.2, ncols=3, nrows=1)

    # Panel (a): 1D thin-layer spontaneous-emission power vs noise_amplitude
    ax = axes1[0]
    amp_dense = np.linspace(0.0, 2.05, 100)
    ax.plot(
        amp_dense,
        (amp_dense**2) * b1["p_sp_theory_W"] * 1e15,
        **figstyle.series_style(0, with_marker=False),
        label=r"1D FDT theory $\alpha^2 f_{\mathrm{rad,1D}} N_2 V \hbar\omega_a / \tau_{21}$",
    )
    ax.plot(
        b1["amps"],
        b1["p_fit_fW"],
        **figstyle.series_style(1, with_line=False, ms=6.2),
        label=f"3D FIT ({b1['n_seeds']}-seed mean at $\\alpha=1$, $\\sigma_{{\\mathrm{{SEM}}}}={b1['sigma_sem_rel']*100:.1f}\\%$)",
    )
    ax.set_xlabel(r"Noise amplitude multiplier $\alpha$ [-]")
    ax.set_ylabel(r"Radiated power $P_{\mathrm{rad}}$ [fW]")
    ax.set_title("Spontaneous-Emission Power (1D FDT)")
    figstyle.spectrum_grid(ax)
    ax.legend(loc="upper left")
    figstyle.panel_label(ax, "(a)")

    # Panel (b): 1D slab gain-narrowed ASE spectrum vs analytic formula
    ax = axes1[1]
    ax.plot(
        b2["f_THz"],
        b2["s_lorentz"],
        **figstyle.series_style(3, with_marker=False),
        label=f"Low-gain Lorentzian ($g_0 L = 0.15$, $\\sigma_f = {b2['rms_th_lo_THz']:.1f}$ THz)",
    )
    ax.plot(
        b2["f_THz"],
        b2["s_an"],
        **figstyle.series_style(0, with_marker=False),
        label=f"Analytic ASE ($g_0 L = 2.80$, $\\sigma_f = {b2['rms_th_hi_THz']:.1f}$ THz)",
    )
    ax.plot(
        b2["f_THz"],
        b2["s_fit"],
        **figstyle.series_style(1, with_marker=True, markevery=6, ms=4.8),
        label=f"3D FIT ASE ($g_0 L = 2.80$, $\\sigma_f = {b2['rms_fit_hi_THz']:.1f}$ THz)",
    )
    ax.set_xlim(175.0, 325.0)
    ax.set_ylim(0.0, 1.15)
    ax.set_xlabel("Frequency $f$ [THz]")
    ax.set_ylabel("Normalized spectral density $S(f) / S_{\\max}$ [-]")
    ax.set_title("1D Slab Gain-Narrowed ASE Spectrum")
    figstyle.spectrum_grid(ax)
    ax.legend(loc="upper right")
    figstyle.panel_label(ax, "(b)")

    # Panel (c): Fabry-Perot L-I curve from spontaneous-emission noise with beta rounding
    ax = axes1[2]
    ax.plot(
        b34["r_dense"],
        b34["i_an_sharp_MW_m2"],
        **figstyle.series_style(3, with_marker=False),
        label=r"Rate eq. ($\beta_{\mathrm{sp}} = 0$, sharp threshold)",
    )
    ax.plot(
        b34["r_dense"],
        b34["i_an_apriori_MW_m2"],
        **figstyle.series_style(2, with_marker=False),
        label=f"A priori theory ($\\beta_{{\\mathrm{{sp,th}}}} = {b34['beta_theory']:.4f}$)",
    )
    ax.plot(
        b34["r_dense"],
        b34["i_an_beta_MW_m2"],
        **figstyle.series_style(0, with_marker=False),
        label=f"Threshold fit ($\\beta_{{\\mathrm{{sp,fit}}}} = {b34['beta_eff']:.4f}$)",
    )
    ax.plot(
        b34["r_sweep"],
        b34["i_fit_MW_m2"],
        **figstyle.series_style(1, with_line=False, ms=6.2),
        label="3D FIT (FDT noise $\\alpha=1$, no seed)",
    )
    ax.axvline(1.0, color=figstyle.PALETTE[2], linestyle=":", linewidth=1.2)
    ax.set_xlabel(r"Normalized pump rate $r = W_p / W_{p,\mathrm{th}}$ [-]")
    ax.set_ylabel(r"Output intensity $I_{\mathrm{out}}$ [MW/m$^2$]")
    ax.set_title(r"Lasing from Noise & $\beta_{\mathrm{sp}}$ Threshold Rounding")
    figstyle.spectrum_grid(ax)
    ax.legend(loc="upper left")
    figstyle.panel_label(ax, "(c)")

    fig1_path = IMG_DIR / "laser_spontaneous_emission_and_ase.png"
    figstyle.save(fig1, fig1_path)
    plt.close(fig1)

    # ------------------------------------------------------------------
    # Figure 2: Multi-mode homogeneous mode competition and lifetime
    #           acceleration (s = 1, 10, 100 in Class-B; s = 1000 breakdown)
    # ------------------------------------------------------------------
    fig2, axes2 = figstyle.new_figure(width_in=14.0, height_in=4.2, ncols=3, nrows=1)

    # Panel (a): Multi-mode cavity mode competition (early multi-mode vs late single-mode)
    ax = axes2[0]
    spec_early_dB = 10.0 * np.log10(np.maximum(b34["spec_early"], 1e-6))
    spec_late_dB = 10.0 * np.log10(np.maximum(b34["spec_late"], 1e-6))
    ax.plot(
        b34["f_mm_THz"],
        spec_early_dB,
        **figstyle.series_style(3, with_marker=False),
        label="Early build-up ($0.05$–$0.45$ ps, multi-mode)",
    )
    ax.plot(
        b34["f_mm_THz"],
        spec_late_dB,
        **figstyle.series_style(0, with_marker=False),
        label=f"Steady state ($1.5$–$2.4$ ps, SMSR $= {b34['smsr_dB']:.1f}$ dB)",
    )
    ax.set_xlim(198.0, 302.0)
    ax.set_ylim(-55.0, 8.0)
    ax.set_xlabel("Frequency $f$ [THz]")
    ax.set_ylabel("Normalized mode power [dB]")
    ax.set_title("Multi-Mode Cavity Competition ($L_{\\mathrm{cav}}=4.8\\,\\mu$m)")
    figstyle.spectrum_grid(ax)
    ax.legend(loc="upper right")
    figstyle.panel_label(ax, "(a)")

    # Panel (b): Lifetime acceleration turn-on delay collapse & steady-state output invariance
    ax = axes2[1]
    for idx, s in enumerate(b5["scales"]):
        t_scaled, dn_norm = b5["turn_on_traces"][s]
        ax.plot(
            t_scaled,
            dn_norm,
            **figstyle.series_style(
                idx, with_marker=True, markevery=max(len(t_scaled) // 8, 1), ms=4.8
            ),
            label=f"$s = {int(s)}$ ($I_{{\\mathrm{{out}}}} = {b5['i_outs_MW_m2'][idx]:.2f}$ MW/m$^2$)",
        )
    ax.axhline(
        1.0,
        color=figstyle.INK,
        linestyle=":",
        linewidth=1.1,
        label="Threshold $\\Delta N_{\\mathrm{th}}$",
    )
    ax.set_xlabel(r"Rescaled time $s \cdot t$ [ps] ($t_{\mathrm{th}}(s) \propto 1/s$)")
    ax.set_ylabel(r"Normalized inversion $\Delta N(t) / \Delta N_{\mathrm{th}}$ [-]")
    ax.set_title("Turn-On Collapse & Steady-State Invariance")
    figstyle.spectrum_grid(ax)
    ax.legend(loc="lower right")
    figstyle.panel_label(ax, "(b)")

    # Panel (c): Relaxation-oscillation scaling f_RO(s) ~ sqrt(s) for s <= 100 and Class-B limit
    ax = axes2[2]
    s_arr = np.array(b5["scales"], dtype=float)
    s_max_cav = b5["limits_by_s"]["10"]["s_max_cavity"]
    s_class_b = np.logspace(0, np.log10(0.95 * s_max_cav), 100)
    t_q_undamped_1 = b5["t_quarter_undamped_disp_fs"][0]
    t_q_disp_curve = []
    for sv in s_class_b:
        m_sv = FourLevelGain(**b5["ref_mat_params"], time_scale=float(sv))
        f_sv = m_sv.relaxation_oscillation_frequency(
            b5["r_pump"], b5["tau_p_disp"], damped=True
        )
        t_q_disp_curve.append(0.25 / f_sv * 1e15 if f_sv > 0.0 else np.nan)
    ax.loglog(
        s_class_b,
        t_q_undamped_1 / np.sqrt(s_class_b),
        **figstyle.series_style(0, with_marker=False),
        label=r"Undamped $\sqrt{s}$: $t_{1/4,\mathrm{disp},0}(1) / \sqrt{s}$",
    )
    ax.loglog(
        s_class_b,
        np.array(t_q_disp_curve),
        **figstyle.series_style(3, with_marker=False),
        label=r"Damped 2-ODE theory $t_{1/4,\mathrm{disp}}(s)$",
    )
    ax.loglog(
        s_arr[:3],
        b5["t_quarter_fs"][:3],
        **figstyle.series_style(1, with_line=False, ms=6.5),
        label=r"3D FIT $t_{\mathrm{zc}}(s)$ ($s \leq 100$)",
    )
    ax.loglog(
        s_arr[3:],
        b5["t_quarter_fs"][3:],
        marker="s",
        linestyle="none",
        markersize=6.8,
        color=figstyle.PALETTE[3],
        label=r"3D FIT $t_{\mathrm{zc}}(1000)$ ($s > s_{\max,\mathrm{cav}}$)",
    )
    ax.loglog(
        s_arr,
        np.array(b5["t_th_num_ps"]) * 1e3,
        **figstyle.series_style(2, with_line=True, ms=5.8),
        label=r"Turn-on delay $t_{\mathrm{th}}(s) \propto 1/s$",
    )
    ax.axvline(
        s_max_cav,
        color=figstyle.PALETTE[3],
        linestyle="--",
        linewidth=1.1,
        label=f"$s_{{\\max,\\mathrm{{cav}}}} = {s_max_cav:.0f}$",
    )
    ax.set_xlabel(r"Lifetime acceleration factor $s$ (time_scale) [-]")
    ax.set_ylabel("Characteristic time scale [fs]")
    ax.set_title(r"Time-Scaling Laws & Class-B Limit ($s_{\max,\mathrm{cav}}=677$)")
    figstyle.spectrum_grid(ax)
    ax.legend(loc="upper right")
    figstyle.panel_label(ax, "(c)")

    fig2_path = IMG_DIR / "laser_mode_competition_and_time_scaling.png"
    figstyle.save(fig2, fig2_path)
    plt.close(fig2)
    return fig1_path, fig2_path


def main():
    t0 = time.perf_counter()
    print("=== Example 014: Spontaneous-Emission Noise & Lifetime Acceleration ===")
    b1 = run_spontaneous_decay_benchmark()
    print(
        f"[1] 1D FDT spontaneous power ({b1['n_seeds']}-seed ensemble): "
        f"FIT = {b1['p_sp_fit_W']*1e15:.4f} fW vs theory = {b1['p_sp_theory_W']*1e15:.4f} fW "
        f"(rel. err = {b1['rel_err_fdt']*100:.2f}%, sigma_SEM = {b1['sigma_sem_rel']*100:.2f}%, "
        f"sigma_1 = {b1['sigma_single_seed_rel']*100:.2f}%, f_rad,1D = {b1['f_rad_1d']:.6f})"
    )

    b2 = run_ase_spectrum_benchmark()
    print(
        f"[2] 1D ASE gain narrowing (g0*L = {b2['g0_L']:.2f}): "
        f"FWHM analytic = {b2['fwhm_an_THz']:.2f} THz (ideal Lorentzian = {b2['fwhm_ideal_lorentz_THz']:.2f} THz), "
        f"RMS width FIT = {b2['rms_fit_hi_THz']:.2f} THz vs analytic = {b2['rms_th_hi_THz']:.2f} THz "
        f"(rel. err = {b2['rel_err_rms_hi']*100:.2f}%, low-gain rel. err = {b2['rel_err_rms_lo']*100:.2f}%)"
    )

    b34 = run_fp_noise_and_mode_competition()
    print(
        f"[3] FP laser from noise (alpha=1): I_out(noise=0) = {b34['i_zero_noise']:.2e} W/m^2, "
        f"beta_theory = {b34['beta_theory']:.5f}, beta_fit = {b34['beta_eff']:.5f} "
        f"(diff = {b34['rel_diff_beta_apriori_vs_eff']*100:.1f}%), "
        f"sub-threshold r<=1 L2 err (a priori) = {b34['l2_err_sub_th_apriori']*100:.2f}% "
        f"(vs sharp beta=0: {b34['l2_err_sub_th_sharp']*100:.1f}%), "
        f"r=1.0 rel. err = {b34['rel_err_r1_apriori']*100:.2f}%, "
        f"full-range L2 (insensitive, dominated by r>=1.5): a priori = {b34['l2_err_beta_apriori']*100:.2f}%, "
        f"sharp = {b34['l2_err_beta_sharp']*100:.2f}%"
    )
    print(
        f"[4] Multi-mode competition (L_cav = 4.8 um): winning mode f = {b34['f_peak_mm_THz']:.2f} THz, "
        f"dominant mode fraction = {b34['dominant_mode_fraction']*100:.1f}%, SMSR = {b34['smsr_dB']:.1f} dB"
    )

    b5 = run_lifetime_acceleration_benchmark()
    print(
        f"[5] Lifetime acceleration s = {b5['scales']}: "
        f"cold ringdown tau_p,cold = {b5['tau_p_cold_fs']:.2f} fs (Q = {b5['Q_cold']:.2f}), "
        f"I_out = {[round(v, 3) for v in b5['i_outs_MW_m2']]} MW/m^2 "
        f"(theory = {b5['i_out_theory_MW_m2']:.3f} MW/m^2), "
        f"cold-start s={b5['cold_start_scales']} I_out = {[round(v, 3) for v in b5['cold_start_i_outs_MW_m2']]} MW/m^2, "
        f"t_zc(RO, FIT) = {[round(v, 2) for v in b5['t_quarter_fs']]} fs "
        f"vs damped disp theory = {[round(v, 2) for v in b5['t_quarter_disp_fs']]} fs "
        f"(rel diff s=1,10,100: {[round(v*100, 2) for v in b5['rel_err_ro_vs_disp_damped']]}%)"
    )

    f1, f2 = make_figures(b1, b2, b34, b5)
    elapsed = time.perf_counter() - t0

    summary = {
        "grid_dz_nm": 25.0,
        "cells_per_wavelength_vacuum": 48.0,
        "cells_per_wavelength_n_H": 24.0,
        "spontaneous_decay_1d": {
            "f_rad_1d": b1["f_rad_1d"],
            "n_seeds": b1["n_seeds"],
            "p_sp_fit_W": b1["p_sp_fit_W"],
            "p_sp_single_seed42_W": b1["p_sp_single_seed42_W"],
            "p_sp_seeds_fW": b1["p_sp_seeds_fW"],
            "sigma_single_seed_rel": b1["sigma_single_seed_rel"],
            "sigma_sem_rel": b1["sigma_sem_rel"],
            "p_sp_theory_W": b1["p_sp_theory_W"],
            "rel_err_fdt": b1["rel_err_fdt"],
            "rel_err_seed42": b1["rel_err_seed42"],
            "noise_amps": b1["amps"].tolist(),
            "p_fit_fW": b1["p_fit_fW"].tolist(),
            "rel_err_quad_alpha2": b1["rel_err_quad_2"],
            "rel_err_quad_alpha05": b1["rel_err_quad_05"],
        },
        "ase_1d_slab": {
            "g0_L": b2["g0_L"],
            "fwhm_homogeneous_THz": b2["fwhm_hom_THz"],
            "fwhm_analytic_THz": b2["fwhm_an_THz"],
            "fwhm_ideal_lorentz_THz": b2["fwhm_ideal_lorentz_THz"],
            "rms_width_fit_hi_THz": b2["rms_fit_hi_THz"],
            "rms_width_theory_hi_THz": b2["rms_th_hi_THz"],
            "rms_width_fit_lo_THz": b2["rms_fit_lo_THz"],
            "rms_width_theory_lo_THz": b2["rms_th_lo_THz"],
            "rel_err_rms_hi": b2["rel_err_rms_hi"],
            "rel_err_rms_lo": b2["rel_err_rms_lo"],
        },
        "fabry_perot_from_noise": {
            "i_out_zero_noise_W_m2": b34["i_zero_noise"],
            "beta_sp_theory": b34["beta_theory"],
            "beta_sp_eff": b34["beta_eff"],
            "rel_diff_beta_apriori_vs_eff": b34["rel_diff_beta_apriori_vs_eff"],
            "f_rad_1d_fp": b34["f_rad_fp"],
            "gamma_z_fp": b34["gamma_z_fp"],
            "spectral_overlap_fp": b34["spectral_overlap_fp"],
            "r_sweep": b34["r_sweep"].tolist(),
            "i_fit_MW_m2": b34["i_fit_MW_m2"].tolist(),
            "l2_err_sub_th_apriori": b34["l2_err_sub_th_apriori"],
            "l2_err_sub_th_eff": b34["l2_err_sub_th_eff"],
            "l2_err_sub_th_sharp": b34["l2_err_sub_th_sharp"],
            "rel_err_r1_apriori": b34["rel_err_r1_apriori"],
            "rel_err_r085_apriori": b34["rel_err_r085_apriori"],
            "l2_err_beta_curve": b34["l2_err_beta"],
            "l2_err_beta_oos": b34["l2_err_beta_oos"],
            "l2_err_beta_apriori": b34["l2_err_beta_apriori"],
            "l2_err_beta_sharp": b34["l2_err_beta_sharp"],
        },
        "multimode_competition": {
            "L_cav_um": 4.8,
            "winning_mode_freq_THz": b34["f_peak_mm_THz"],
            "dominant_mode_power_fraction": b34["dominant_mode_fraction"],
            "side_mode_early_norm": b34["side_early"],
            "side_mode_late_norm": b34["side_late"],
            "smsr_dB": b34["smsr_dB"],
        },
        "lifetime_acceleration": {
            "scales": b5["scales"],
            "class_b_valid_scales": [1.0, 10.0, 100.0],
            "f_cold_THz": b5["f_cold_THz"],
            "Q_cold": b5["Q_cold"],
            "kappa_cav_s_inv": b5["kappa_cav_s_inv"],
            "tau_p_cold_fs": b5["tau_p_cold_fs"],
            "tau_p_disp_fs": b5["tau_p_disp_fs"],
            "t_th_theory_ps": b5["t_th_theory_ps"],
            "t_th_numerical_ps": b5["t_th_num_ps"],
            "t_th_rel_err": b5["t_th_rel_err"],
            "f_ro_theory_THz": b5["f_ro_theory_THz"],
            "f_ro_undamped_disp_THz": b5["f_ro_undamped_disp_THz"],
            "f_ro_damped_bare_THz": b5["f_ro_damped_bare_THz"],
            "f_ro_damped_disp_THz": b5["f_ro_damped_disp_THz"],
            "t_quarter_ro_bare_theory_fs": b5["t_quarter_bare_fs"],
            "t_quarter_ro_disp_theory_fs": b5["t_quarter_disp_fs"],
            "t_quarter_ro_undamped_disp_theory_fs": b5["t_quarter_undamped_disp_fs"],
            "rel_err_ro_vs_disp_damped": b5["rel_err_ro_vs_disp_damped"],
            "rel_err_ro_vs_disp_undamped": b5["rel_err_ro_vs_disp_undamped"],
            "ratio_t_quarter_1_to_10_fit": b5["ratio_t_quarter_1_to_10_fit"],
            "ratio_t_quarter_1_to_100_fit": b5["ratio_t_quarter_1_to_100_fit"],
            "ratio_t_quarter_1_to_10_disp_damped": b5[
                "ratio_t_quarter_1_to_10_disp_damped"
            ],
            "ratio_t_quarter_1_to_100_disp_damped": b5[
                "ratio_t_quarter_1_to_100_disp_damped"
            ],
            "i_outs_MW_m2": b5["i_outs_MW_m2"],
            "i_out_theory_MW_m2": b5["i_out_theory_MW_m2"],
            "dn_ss_m3": b5["dn_ss_list"],
            "f_peaks_THz": b5["f_peaks_THz"],
            "t_quarter_ro_fs": b5["t_quarter_fs"],
            "cold_start_scales": b5["cold_start_scales"],
            "cold_start_i_outs_MW_m2": b5["cold_start_i_outs_MW_m2"],
            "cold_start_i_out_theory_MW_m2": b5["cold_start_i_out_theory_MW_m2"],
            "cold_start_dn_ss_m3": b5["cold_start_dn_ss_m3"],
            "limits_by_s": b5["limits_by_s"],
        },
        "figures": [str(f1.relative_to(REPO_ROOT)), str(f2.relative_to(REPO_ROOT))],
        "wall_time_s": elapsed,
    }
    JSON_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Saved JSON metrics to {JSON_PATH}")
    print(f"Saved figures:\n  {f1}\n  {f2}")
    print(f"Completed in {elapsed:.1f} s.")


if __name__ == "__main__":
    main()
