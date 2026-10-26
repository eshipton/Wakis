# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Example 015 — Semiconductor Gain Medium and 1D DBR Fabry–Pérot Diode Laser
===========================================================================

Demonstrates the `SemiconductorGain` active material model (`wakis.gain_semiconductor`)
coupled to `SolverFIT3D` across five physical benchmarks from semiconductor diode-laser
theory (L. A. Coldren, S. W. Corzine, and M. L. Mašanović, *Diode Lasers and Photonic
Integrated Circuits*, 2nd ed., Wiley, 2012, doi:10.1002/9781118148167; C. H. Henry,
*IEEE J. Quantum Electron.* **18**(2), 259–264, 1982, doi:10.1109/JQE.1982.1071522;
T. L. Koch and J. E. Bowers, "Nature of wavelength chirping in directly modulated
semiconductor lasers", *Electron. Lett.* **20**(25), 1038–1040, 1984,
doi:10.1049/el:19840709):

1. **Small-signal gain spectrum**: 3D FIT transmission through a pumped semiconductor
   slab vs analytical Lorentzian susceptibility ``chi(omega, N)`` for both linear
   ``g'(N - N_tr)`` and logarithmic ``g_0 ln((N + N_s)/(N_tr + N_s))`` gain models.
2. **Henry linewidth enhancement factor**: 3D FIT complex slab transmission phase-to-gain
   ratio ``Delta phi / Delta ln|t|`` under carrier density variation ``Delta N`` vs
   input ``alpha_H``.
3. **1D DBR Fabry–Pérot diode laser L–I curve**: 3D FIT steady-state output intensity
   and carrier clamping at ``N_th`` across threshold ``J_th``, verifying differential
   quantum slope efficiency ``eta_d = eta_i * alpha_m / (alpha_i + alpha_m)``.
4. **Modulation response, relaxation oscillations, and adiabatic chirp**: small-signal
   transfer function ``H(omega_m)``, relaxation-oscillation frequency
   ``omega_R = sqrt(v_g * g' * S_0 / tau_p)`` (in 3D FIT and coupled rate equations),
   and Koch–Bowers frequency chirp ``Delta nu(t) = (alpha_H / (4 pi)) d(ln P)/dt``.
5. **Carrier–photon number conservation and carrier diffusion**: exact particle budget
   ``Delta N_removed = N_photons = Delta W_EM / (hbar * omega_0)`` in a closed 3D FIT
   cavity and spatial carrier diffusion decay ``exp(-D k^2 t)``.

Saves two publication-style figures in ``docs/img/`` and writes all benchmark metrics
to ``examples/data/015_semiconductor_fp_laser.json``:
  1. ``docs/img/semiconductor_gain_spectrum_and_alpha.png``
  2. ``docs/img/semiconductor_fp_laser_li_and_chirp.png``
"""

from __future__ import annotations

import json
import logging
import os
import time

import matplotlib

matplotlib.use("Agg")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c, e as q_e, epsilon_0, hbar, mu_0

import wakis
from wakis.estimate import estimate, preflight
from wakis.figstyle import (
    PALETTE,
    apply_style,
    new_figure,
    panel_label,
    save,
    spectrum_grid,
)
from wakis.gain import dbr_reflectance, fabry_perot_threshold_gain
from wakis.gain_semiconductor import SemiconductorGain, adiabatic_chirp
from wakis.materials import DispersiveMaterial
from wakis.resonances import ResonanceMonitor
from wakis.sources import ContinuousSource, CurrentSource, GaussianSource


def preflight_1d_fit(nz, steps, n_gain_cells=0, n_poles=1):
    """Run `wakis.estimate.estimate` and `preflight` against the laptop 2 GB RAM cap."""
    est = estimate(
        (2, 2, int(nz)),
        steps=int(steps),
        dtype="float64",
        backend="cpu",
        cpml=True,
        dispersive_edges=3 * int(n_gain_cells),
        n_poles=int(n_poles),
    )
    problems = preflight(est, "laptop", raise_error=True)
    if problems:
        raise RuntimeError(f"Preflight check failed: {problems}")
    return est


# ---------------------------------------------------------------------------
# 1. Small-signal gain spectrum g(omega, N) and Henry alpha_H slab transmission
# ---------------------------------------------------------------------------


def run_slab_gain_and_alpha():
    """Run 3D FIT slab transmission benchmarks for g(omega, N) and Henry alpha_H."""
    lam0 = 1.55e-6
    f0 = c / lam0  # 193.414 THz
    omega_0 = 2.0 * np.pi * f0
    gamma = 2.0 * np.pi * 25.0e12
    dz = 20.0e-9  # 77.5 cells/lambda0
    L_slab = 1.2e-6  # 60 cells

    N_tr = 1.0e24
    g_prime = 8.0e-20  # [m^2]
    g0_log = g_prime * N_tr

    z_min, z_max = -1.6e-6, 1.6e-6
    Nz = int(round((z_max - z_min) / dz))
    grid = wakis.GridFIT3D(
        -1.0e-6, 1.0e-6, -1.0e-6, 1.0e-6, z_min, z_max, 2, 2, Nz, verbose=0
    )
    f_band = np.linspace(f0 - 18.0e12, f0 + 18.0e12, 33)

    def run_broadband(material=None):
        solver = wakis.SolverFIT3D(
            grid,
            cfln=0.5,
            bc_low=["periodic", "periodic", "cpml"],
            bc_high=["periodic", "periodic", "cpml"],
            n_pml=10,
            verbose=0,
        )
        if material is not None:
            solver.add_gain_material(
                material, center=(0.0, 0.0, 0.0), size=(4.0e-6, 4.0e-6, L_slab)
            )
        src = GaussianSource(frequency=f0, fwidth=28.0e12)
        sheet = CurrentSource(
            src,
            component="x",
            center=(0.0, 0.0, -1.0e-6),
            size=(2.0e-6, 2.0e-6, 0.0),
            amplitude=1.0e-3,
        )
        flux_mon = wakis.FluxMonitor(
            f_band, center=(0.0, 0.0, 1.0e-6), size=(1.0e-6, 1.0e-6, 0.0)
        )
        Nt = int((src.end_time + 220.0e-15) / solver.dt)
        preflight_1d_fit(Nz, Nt, n_gain_cells=int(L_slab / dz))
        solver.emsolve(Nt, source=sheet, monitors=[flux_mon])
        return flux_mon.get_flux(), solver.dt

    phi_vac, dt_fit = run_broadband(None)

    cases_spec = {}
    for label, model_name, n_val in [
        ("abs_0.25Ntr", "linear", 0.25 * N_tr),
        ("trans_1.0Ntr", "linear", 1.00 * N_tr),
        ("gain_2.5Ntr", "linear", 2.50 * N_tr),
        ("log_2.5Ntr", "logarithmic", 2.50 * N_tr),
    ]:
        mat = SemiconductorGain(
            omega_0=omega_0,
            gamma=gamma,
            N_tr=N_tr,
            differential_gain=g_prime,
            g0=g0_log,
            gain_model=model_name,
            alpha_H=0.0,
            A=0.0,
            B=0.0,
            C=0.0,
            eps_inf=1.0,
            initial_carriers=n_val,
            enable_stimulated=False,
        )
        phi_out, _ = run_broadband(mat)
        g_fit = np.log(phi_out / phi_vac) / L_slab
        g_an = mat.small_signal_gain(f=f_band, N=n_val, dt=dt_fit)
        n_c = mat.refractive_index(f=f_band, N=n_val, dt=dt_fit)
        k_z = (2.0 * np.pi * f_band / c) * n_c
        k_0 = 2.0 * np.pi * f_band / c
        t_slab = (4.0 * n_c * np.exp(-1j * (k_z - k_0) * L_slab)) / (
            (n_c + 1.0) ** 2 - (n_c - 1.0) ** 2 * np.exp(-2j * k_z * L_slab)
        )
        t2_ratio = phi_out / phi_vac
        t2_exact = np.abs(t_slab) ** 2
        idx_c = len(f_band) // 2
        rel_err_t2 = float(
            np.max(np.abs(t2_ratio - t2_exact)) / np.max(t2_exact)
        )
        rel_err_center = (
            float(abs(g_fit[idx_c] - g_an[idx_c]) / abs(g_an[idx_c]))
            if abs(g_an[idx_c]) > 1.0
            else float(abs(g_fit[idx_c]))
        )
        cases_spec[label] = {
            "g_fit": g_fit,
            "g_an": g_an,
            "g_center_fit": float(g_fit[idx_c]),
            "g_center_an": float(g_an[idx_c]),
            "rel_err_t2": rel_err_t2,
            "rel_err_center": rel_err_center,
        }

    # Henry alpha_H complex transmission measurement at f0
    def measure_cw_phasor(material):
        solver = wakis.SolverFIT3D(
            grid,
            cfln=0.5,
            bc_low=["periodic", "periodic", "cpml"],
            bc_high=["periodic", "periodic", "cpml"],
            n_pml=10,
            verbose=0,
        )
        solver.add_gain_material(
            material, center=(0.0, 0.0, 0.0), size=(4.0e-6, 4.0e-6, L_slab)
        )
        src = GaussianSource(frequency=f0, fwidth=0.12 * f0)
        sheet = CurrentSource(
            src,
            component="x",
            center=(0.0, 0.0, -1.0e-6),
            size=(2.0e-6, 2.0e-6, 0.0),
            amplitude=1.0e-3,
        )
        mon = wakis.DFTMonitor([f0], components=["Ex"], center=(0.0, 0.0, 1.0e-6))
        Nt = int((src.end_time + 120.0e-15) / solver.dt)
        preflight_1d_fit(Nz, Nt, n_gain_cells=int(L_slab / dz))
        solver.emsolve(Nt, source=sheet, monitors=[mon])
        return complex(mon.get_field("Ex")[0][0, 0, 0, 0])

    mat_tr_cw = SemiconductorGain(
        omega_0=omega_0,
        gamma=gamma,
        N_tr=N_tr,
        differential_gain=2.5e-20,
        alpha_H=0.0,
        A=0.0,
        eps_inf=1.0,
        initial_carriers=N_tr,
        enable_stimulated=False,
    )
    u_ref = measure_cw_phasor(mat_tr_cw)

    alpha_results = {}
    for alpha_in in (2.0, 4.0, 5.0):
        dln_list = []
        dphi_list = []
        for mult in (1.15, 1.30, 1.45):
            mat_a = SemiconductorGain(
                omega_0=omega_0,
                gamma=gamma,
                N_tr=N_tr,
                differential_gain=2.5e-20,
                alpha_H=alpha_in,
                A=0.0,
                eps_inf=1.0,
                initial_carriers=mult * N_tr,
                enable_stimulated=False,
            )
            ratio = measure_cw_phasor(mat_a) / u_ref
            dln_list.append(float(np.log(np.abs(ratio))))
            dphi_list.append(float(np.angle(ratio)))
        alpha_meas = float(dphi_list[1] / dln_list[1])
        alpha_results[f"alpha_{alpha_in:.1f}"] = {
            "alpha_input": alpha_in,
            "alpha_measured": alpha_meas,
            "rel_err": float(abs(alpha_meas - alpha_in) / alpha_in),
            "dln_amp": dln_list,
            "dphi_rad": dphi_list,
        }

    return {
        "f_THz": f_band * 1e-12,
        "cases": cases_spec,
        "alpha": alpha_results,
    }


# ---------------------------------------------------------------------------
# 2. 1D DBR Fabry-Perot diode laser L-I curve and slope efficiency
# ---------------------------------------------------------------------------


def run_fp_diode_laser_li():
    """Run 3D FIT 1D DBR Fabry-Perot diode laser L-I sweep across threshold."""
    lam0 = 1.2e-6
    f0 = c / lam0  # 249.827 THz
    omega_0 = 2.0 * np.pi * f0
    gamma = 2.0 * np.pi * 40.0e12
    dz = 25.0e-9  # 48 cells/lambda in cavity, 24 cells/lambda in n_H = 2.0

    n_H, n_L = 2.0, 1.0
    d_H = lam0 / (4.0 * n_H)  # 150 nm = 6 dz
    d_L = lam0 / (4.0 * n_L)  # 300 nm = 12 dz
    # Symmetric 3-layer (H L H) quarter-wave DBR mirror on each side:
    # n_pairs = 1 (H, L) pair plus 1 extra H layer (extra_high=True), R = (15/17)^2 = 0.77855
    n_pairs = 1
    R_mirror = dbr_reflectance(n_H, n_L, n_pairs, n_in=1.0, n_out=1.0, extra_high=True)
    L_cav = 1.2e-6
    alpha_m = np.log(1.0 / R_mirror) / L_cav

    N_tr = 5.0e19
    g_prime = 1.8e-14
    A_rate = 1.0 / 150.0e-15
    eta_i = 0.85
    d_act = L_cav

    semi_ref = SemiconductorGain(
        omega_0=omega_0,
        gamma=gamma,
        N_tr=N_tr,
        differential_gain=g_prime,
        alpha_H=0.0,
        eta_i=eta_i,
        active_thickness=d_act,
        A=A_rate,
        eps_inf=1.0,
    )
    ss_ref = semi_ref.steady_state_laser(
        J=0.0, length=L_cav, R1=R_mirror, R2=R_mirror
    )
    J_th = ss_ref["J_th"]
    N_th = ss_ref["N_th"]
    J_tr = (q_e * d_act / eta_i) * A_rate * N_tr

    z_min, z_max = -1.2e-6, 2.4e-6
    Nz = int(round((z_max - z_min) / dz))
    grid = wakis.GridFIT3D(
        -1.0e-6, 1.0e-6, -1.0e-6, 1.0e-6, z_min, z_max, 2, 2, Nz, verbose=0
    )

    def run_point(J_val, total_time=0.9e-12):
        solver = wakis.SolverFIT3D(
            grid,
            cfln=0.8,
            bc_low=["periodic", "periodic", "cpml"],
            bc_high=["periodic", "periodic", "cpml"],
            n_pml=10,
            verbose=0,
        )
        mat_H = DispersiveMaterial(eps_inf=n_H**2, poles=[])
        for m in range(n_pairs + 1):
            z_left = -(m * (d_H + d_L) + 0.5 * d_H)
            solver.add_dispersive_material(
                mat_H, center=(0.0, 0.0, z_left), size=(4.0e-6, 4.0e-6, d_H)
            )
            z_right = L_cav + m * (d_H + d_L) + 0.5 * d_H
            solver.add_dispersive_material(
                mat_H, center=(0.0, 0.0, z_right), size=(4.0e-6, 4.0e-6, d_H)
            )
        mat = SemiconductorGain(
            omega_0=omega_0,
            gamma=gamma,
            N_tr=N_tr,
            differential_gain=g_prime,
            alpha_H=0.0,
            J=J_val,
            eta_i=eta_i,
            active_thickness=d_act,
            A=A_rate,
            eps_inf=1.0,
            initial_carriers="steady_state" if J_val > 0 else N_tr,
            enable_stimulated=True,
            spatial_averaging=True,
        )
        solver.add_gain_material(
            mat, center=(0.0, 0.0, 0.5 * L_cav), size=(4.0e-6, 4.0e-6, L_cav)
        )
        src = GaussianSource(frequency=f0, fwidth=0.08 * f0)
        sheet = CurrentSource(
            src,
            component="x",
            center=(0.0, 0.0, 0.25 * L_cav),
            size=(2.0e-6, 2.0e-6, 0.0),
            amplitude=2.0,
        )
        z_out = L_cav + 2.0 * d_H + d_L + 0.2e-6
        k_out = int(np.argmin(np.abs(solver.z - z_out)))
        idx_out = k_out * solver.Nx * solver.Ny
        Nt = int(total_time / solver.dt)
        preflight_1d_fit(Nz, Nt, n_gain_cells=int(L_cav / dz))

        ex_out = np.empty(Nt)
        n_hist = np.empty(Nt)
        for n in range(Nt):
            sheet.update(solver, n * solver.dt)
            solver.one_step()
            ex_out[n] = solver.E.array[idx_out]
            n_hist[n] = solver.get_gain_carriers(cell_average=True)
        n_avg = int((20.0 / f0) / solver.dt)
        I_out_1 = c * epsilon_0 * float(np.mean(ex_out[-n_avg:] ** 2))
        return I_out_1, n_hist, solver.dt

    r_excess = np.array([0.0, 0.4, 1.5, 2.0, 2.5, 3.0])
    J_sweep = J_tr + r_excess * (J_th - J_tr)
    I_fit_1 = []
    I_an_1 = []
    N_clamped = []

    for J_val in J_sweep:
        i1, n_hist, _ = run_point(J_val)
        ss_pt = semi_ref.steady_state_laser(
            J=J_val, length=L_cav, R1=R_mirror, R2=R_mirror
        )
        I_fit_1.append(i1)
        I_an_1.append(ss_pt["I_out_1"])
        N_clamped.append(float(n_hist[-1]))

    I_fit_1 = np.array(I_fit_1)
    I_an_1 = np.array(I_an_1)
    N_clamped = np.array(N_clamped)

    above = r_excess > 1.0
    slope_fit, intercept_fit = np.polyfit(J_sweep[above], 2.0 * I_fit_1[above], 1)
    J_th_fit = float(-intercept_fit / slope_fit)
    eta_d_fit = float(slope_fit * (q_e / (hbar * omega_0)) * (d_act / L_cav))

    # Step-bias ringdown in 3D FIT to measure relaxation-oscillation frequency f_R
    J_step = J_tr + 2.5 * (J_th - J_tr)
    _, n_ring, dt_ring = run_point(J_step, total_time=0.9e-12)
    t_ring = np.arange(len(n_ring)) * dt_ring
    mask_ro = (t_ring >= 0.22e-12) & (t_ring <= 0.65e-12)
    dn_ro = n_ring[mask_ro] - N_th
    sign_ch = np.where(np.diff(np.sign(dn_ro)) != 0)[0]
    t_zc = t_ring[mask_ro][sign_ch]
    half_periods = np.diff(t_zc[-4:])
    f_R_fit_THz = float(1.0 / (2.0 * np.mean(half_periods)) * 1e-12)

    # Cold-cavity ringdown (gain off) on the identical 3-layer (H L H) DBR cavity
    # using ResonanceMonitor + harminv to measure tau_p,cold directly:
    solver_cold = wakis.SolverFIT3D(
        grid,
        cfln=0.8,
        bc_low=["periodic", "periodic", "cpml"],
        bc_high=["periodic", "periodic", "cpml"],
        n_pml=10,
        verbose=0,
    )
    mat_H_cold = DispersiveMaterial(eps_inf=n_H**2, poles=[])
    for m in range(n_pairs + 1):
        z_left = -(m * (d_H + d_L) + 0.5 * d_H)
        solver_cold.add_dispersive_material(
            mat_H_cold, center=(0.0, 0.0, z_left), size=(4.0e-6, 4.0e-6, d_H)
        )
        z_right = L_cav + m * (d_H + d_L) + 0.5 * d_H
        solver_cold.add_dispersive_material(
            mat_H_cold, center=(0.0, 0.0, z_right), size=(4.0e-6, 4.0e-6, d_H)
        )
    src_cold = GaussianSource(frequency=f0, fwidth=0.05 * f0)
    sheet_cold = CurrentSource(
        src_cold,
        component="x",
        center=(0.0, 0.0, 0.25 * L_cav),
        size=(2.0e-6, 2.0e-6, 0.0),
        amplitude=1.0,
    )
    res_mon = ResonanceMonitor(
        component="Ex",
        center=(0.0, 0.0, 0.75 * L_cav),
        t_start=src_cold.end_time + 30.0e-15,
    )
    Nt_cold = int((src_cold.end_time + 300.0e-15) / solver_cold.dt)
    preflight_1d_fit(Nz, Nt_cold, n_gain_cells=0)
    solver_cold.emsolve(Nt_cold, source=sheet_cold, monitors=[res_mon])
    modes_cold = res_mon.harminv(0.85 * f0, 1.15 * f0)
    idx_mode = int(np.argmin(np.abs(modes_cold.frequency - f0)))
    f_cold_THz = float(modes_cold.frequency[idx_mode] * 1e-12)
    Q_cold = float(modes_cold.Q[idx_mode])
    kappa_cold = 2.0 * float(modes_cold.decay_rate[idx_mode])
    tau_p_cold = float(1.0 / kappa_cold)
    L_eff_dbr = float(c * tau_p_cold * np.log(1.0 / R_mirror))
    Gamma_z_dbr = float(L_cav / L_eff_dbr)

    # Also evaluate the asymptotic analytical DBR penetration length L_pen = lambda_0 / (4*(n_H - n_L)):
    L_pen_an = float(lam0 / (4.0 * (n_H - n_L)))
    L_eff_an_dbr = float(L_cav + 2.0 * L_pen_an)
    Gamma_z_an_dbr = float(L_cav / L_eff_an_dbr)

    ss_ro_bare = semi_ref.steady_state_laser(
        J=J_step, length=L_cav, R1=R_mirror, R2=R_mirror
    )
    ss_ro_dbr = semi_ref.steady_state_laser(
        J=J_step, length=L_cav, R1=R_mirror, R2=R_mirror, L_eff=L_eff_dbr
    )
    ss_ro_an_dbr = semi_ref.steady_state_laser(
        J=J_step, length=L_cav, R1=R_mirror, R2=R_mirror, L_eff=L_eff_an_dbr
    )
    f_R_bare_THz = float(ss_ro_bare["f_R"] * 1e-12)
    f_R_an_THz = float(ss_ro_dbr["f_R"] * 1e-12)
    f_R_analytic_lpen_THz = float(ss_ro_an_dbr["f_R"] * 1e-12)

    return {
        "J_tr": float(J_tr),
        "J_th": float(J_th),
        "J_th_fit": J_th_fit,
        "J_th_rel_err": float(abs(J_th_fit - J_th) / J_th),
        "N_tr": float(N_tr),
        "N_th": float(N_th),
        "eta_d_expected": float(ss_ref["eta_d"]),
        "eta_d_fit": eta_d_fit,
        "eta_d_rel_err": float(abs(eta_d_fit - ss_ref["eta_d"]) / ss_ref["eta_d"]),
        "r_excess": r_excess,
        "J_sweep": J_sweep,
        "I_fit_1": I_fit_1,
        "I_an_1": I_an_1,
        "N_clamped": N_clamped,
        "max_rel_err_N_clamped": float(
            np.max(np.abs(N_clamped[above] - N_th) / N_th)
        ),
        "max_rel_err_I_above": float(
            np.max(np.abs(I_fit_1[above] - I_an_1[above]) / I_an_1[above])
        ),
        "f_cold_THz": f_cold_THz,
        "Q_cold": Q_cold,
        "tau_p_cold_fs": float(tau_p_cold * 1e15),
        "L_eff_dbr_m": L_eff_dbr,
        "Gamma_z_dbr": Gamma_z_dbr,
        "L_pen_analytic_m": L_pen_an,
        "L_eff_analytic_dbr_m": L_eff_an_dbr,
        "Gamma_z_analytic_dbr": Gamma_z_an_dbr,
        "f_R_fit_THz": f_R_fit_THz,
        "f_R_bare_THz": f_R_bare_THz,
        "f_R_bare_rel_err": float(abs(f_R_fit_THz - f_R_bare_THz) / f_R_bare_THz),
        "f_R_an_THz": f_R_an_THz,
        "f_R_fit_rel_err": float(abs(f_R_fit_THz - f_R_an_THz) / f_R_an_THz),
        "f_R_analytic_lpen_THz": f_R_analytic_lpen_THz,
        "f_R_analytic_lpen_rel_err": float(
            abs(f_R_fit_THz - f_R_analytic_lpen_THz) / f_R_analytic_lpen_THz
        ),
    }


# ---------------------------------------------------------------------------
# 3. Modulation response H(omega_m), omega_R scaling, and adiabatic chirp
# ---------------------------------------------------------------------------


def run_modulation_and_chirp():
    """Evaluate modulation transfer function, omega_R scaling, and Koch-Bowers chirp."""
    lam0 = 1.55e-6
    f0 = c / lam0
    omega_0 = 2.0 * np.pi * f0
    gamma = 2.0 * np.pi * 30.0e12
    eps_inf = 12.25
    N_tr = 1.2e24
    g_prime = 4.0e-20
    alpha_H = 4.0
    eta_i = 0.90
    d_act = 0.12e-6
    A_srh = 5.0e8
    B_rad = 1.0e-16
    C_aug = 3.0e-41
    L_cav = 300.0e-6
    R_facet = 0.32
    alpha_i = 1000.0

    semi = SemiconductorGain(
        omega_0=omega_0,
        gamma=gamma,
        N_tr=N_tr,
        differential_gain=g_prime,
        alpha_H=alpha_H,
        eta_i=eta_i,
        active_thickness=d_act,
        A=A_srh,
        B=B_rad,
        C=C_aug,
        eps_inf=eps_inf,
        sigma=alpha_i * c * np.sqrt(eps_inf) * epsilon_0,
    )
    ss_th = semi.steady_state_laser(J=0.0, length=L_cav, R1=R_facet, R2=R_facet)
    J_th = ss_th["J_th"]
    N_th = ss_th["N_th"]

    f_mod_GHz = np.linspace(0.1, 20.0, 300)
    bias_factors = [1.5, 2.0, 3.0, 4.0]
    h_curves_dB = {}
    f_r_theory_GHz = []
    f_r_num_GHz = []

    for bf in bias_factors:
        J_b = bf * J_th
        ss_b = semi.steady_state_laser(
            J=J_b, length=L_cav, R1=R_facet, R2=R_facet
        )
        h_tf = semi.modulation_transfer_function(
            f_mod=f_mod_GHz * 1e9,
            J=J_b,
            length=L_cav,
            R1=R_facet,
            R2=R_facet,
        )
        h_curves_dB[f"{bf:.1f}J_th"] = 20.0 * np.log10(np.abs(h_tf))
        f_r_theory_GHz.append(float(ss_b["f_R"] * 1e-9))

        # Numerical step perturbation ringdown
        tr = semi.step_laser_rate_equations(
            Nt=12000,
            dt=0.25e-12,
            J=J_b,
            length=L_cav,
            R1=R_facet,
            R2=R_facet,
            N_init=N_th,
            S_init=1.02 * ss_b["S_ss"],
        )
        t_arr = tr["t"]
        ds = tr["S"] - ss_b["S_ss"]
        zc = np.where(np.diff(np.sign(ds)) != 0)[0]
        t_zc = t_arr[zc[:10]]
        omega_d = np.pi / np.mean(np.diff(t_zc))
        omega_r_num = np.sqrt(omega_d**2 + (0.5 * ss_b["gamma_RO"]) ** 2)
        f_r_num_GHz.append(float(omega_r_num / (2.0 * np.pi) * 1e-9))

    # Dynamic/adiabatic frequency chirp under 2.5 GHz current modulation at J = 2.5 J_th
    # (Standalone ODE consistency check: step_laser_rate_equations vs adiabatic_chirp)
    J_0 = 2.5 * J_th
    ss_0 = semi.steady_state_laser(J=J_0, length=L_cav, R1=R_facet, R2=R_facet)
    f_m = 2.5e9
    omega_m = 2.0 * np.pi * f_m
    delta_J = 0.08 * (J_0 - J_th)
    tr_mod = semi.step_laser_rate_equations(
        Nt=8000,
        dt=0.25e-12,
        J=lambda t: J_0 + delta_J * np.sin(omega_m * t),
        length=L_cav,
        R1=R_facet,
        R2=R_facet,
        N_init=N_th,
        S_init=ss_0["S_ss"],
    )
    t_ns = tr_mod["t"] * 1e9
    dnu_GHz = tr_mod["delta_nu"] * 1e-9
    dnu_formula_GHz = adiabatic_chirp(
        tr_mod["t"], tr_mod["I_out_total"], alpha_H
    ) * 1e-9
    steady = slice(4000, 7900)
    ode_chirp_consistency_rel_err = float(
        np.linalg.norm(dnu_GHz[steady] - dnu_formula_GHz[steady])
        / np.linalg.norm(dnu_GHz[steady])
    )

    # Full-wave 3D FIT Koch-Bowers chirp slope fit under sinusoidal current modulation
    lam_fit = 1.2e-6
    f0_fit = c / lam_fit
    omega_0_fit = 2.0 * np.pi * f0_fit
    gamma_pole = 2.0 * np.pi * 120.0e12
    dz_fit = 25.0e-9
    n_H, n_L = 2.0, 1.0
    d_H = lam_fit / (4.0 * n_H)
    d_L = lam_fit / (4.0 * n_L)
    # Symmetric 5-layer (H L H L H) quarter-wave DBR mirror on each side:
    # n_pairs_fit = 2 (H, L) pairs plus 1 extra H layer (extra_high=True), R = (63/65)^2 = 0.93941
    n_pairs_fit = 2
    R_mirror_fit = dbr_reflectance(
        n_H, n_L, n_pairs_fit, n_in=1.0, n_out=1.0, extra_high=True
    )
    L_cav_fit = 1.2e-6
    N_tr_fit = 5.0e19
    g_prime_fit = 1.8e-14
    A_rate_fit = 1.0 / 200.0e-15
    eta_i_fit = 1.0
    semi_fit_ref = SemiconductorGain(
        omega_0=omega_0_fit,
        gamma=gamma_pole,
        N_tr=N_tr_fit,
        differential_gain=g_prime_fit,
        alpha_H=0.0,
        eta_i=eta_i_fit,
        active_thickness=L_cav_fit,
        A=A_rate_fit,
        eps_inf=1.0,
    )
    ss_fit_th = semi_fit_ref.steady_state_laser(
        J=0.0, length=L_cav_fit, R1=R_mirror_fit, R2=R_mirror_fit
    )
    J_th_fit_mod = ss_fit_th["J_th"]
    N_th_fit_mod = ss_fit_th["N_th"]
    J_tr_fit_mod = (q_e * L_cav_fit / eta_i_fit) * A_rate_fit * N_tr_fit
    J_0_fit = J_tr_fit_mod + 2.5 * (J_th_fit_mod - J_tr_fit_mod)
    f_mod_fit = 1.5e12
    omega_mod_fit = 2.0 * np.pi * f_mod_fit
    delta_J_fit = 0.35 * (J_0_fit - J_th_fit_mod)
    z_min_fit, z_max_fit = -1.5e-6, 2.7e-6
    Nz_fit = int(round((z_max_fit - z_min_fit) / dz_fit))
    grid_fit = wakis.GridFIT3D(
        -1.0e-6, 1.0e-6, -1.0e-6, 1.0e-6, z_min_fit, z_max_fit, 2, 2, Nz_fit, verbose=0
    )

    def run_modulated_3d_fit(alpha_val):
        solver = wakis.SolverFIT3D(
            grid_fit,
            cfln=0.8,
            bc_low=["periodic", "periodic", "cpml"],
            bc_high=["periodic", "periodic", "cpml"],
            n_pml=10,
            verbose=0,
        )
        ieps_arr = solver.ieps.array
        for m in range(n_pairs_fit + 1):
            for z_c in (
                -(m * (d_H + d_L) + 0.5 * d_H),
                L_cav_fit + m * (d_H + d_L) + 0.5 * d_H,
            ):
                idx_e, w_e = solver._box_edges(
                    (0.0, 0.0, z_c), (4.0e-6, 4.0e-6, d_H), dual=False
                )
                eps_s = 1.0 / (epsilon_0 * ieps_arr[idx_e])
                ieps_arr[idx_e] = 1.0 / (
                    epsilon_0 * ((1.0 - w_e) * eps_s + w_e * (n_H**2))
                )
        solver.update_tensors("ieps")
        t_mod_start = 0.9e-12
        j_func = lambda t: (
            J_0_fit
            if t < t_mod_start
            else J_0_fit + delta_J_fit * np.sin(omega_mod_fit * (t - t_mod_start))
        )
        mat = SemiconductorGain(
            omega_0=omega_0_fit,
            gamma=gamma_pole,
            N_tr=N_tr_fit,
            differential_gain=g_prime_fit,
            alpha_H=alpha_val,
            J=j_func,
            eta_i=eta_i_fit,
            active_thickness=L_cav_fit,
            A=A_rate_fit,
            eps_inf=1.0,
            initial_carriers=N_th_fit_mod,
            enable_stimulated=True,
            spatial_averaging=True,
        )
        solver.add_gain_material(
            mat, center=(0.0, 0.0, 0.5 * L_cav_fit), size=(4.0e-6, 4.0e-6, L_cav_fit)
        )
        src = GaussianSource(frequency=f0_fit, fwidth=0.06 * f0_fit)
        sheet = CurrentSource(
            src,
            component="x",
            center=(0.0, 0.0, 0.25 * L_cav_fit),
            size=(2.0e-6, 2.0e-6, 0.0),
            amplitude=2.0,
        )
        z_out = L_cav_fit + (n_pairs_fit + 1) * d_H + n_pairs_fit * d_L + 0.2e-6
        k_out = int(np.argmin(np.abs(solver.z - z_out)))
        idx_out = k_out * solver.Nx * solver.Ny
        Nt = int(1.9e-12 / solver.dt)
        preflight_1d_fit(Nz_fit, Nt, n_gain_cells=int(L_cav_fit / dz_fit))

        ex_out = np.empty(Nt)
        for n in range(Nt):
            sheet.update(solver, n * solver.dt)
            solver.one_step()
            ex_out[n] = solver.E.array[idx_out]

        t_arr = np.arange(Nt) * solver.dt
        fft_e = np.fft.fft(ex_out)
        freqs_e = np.fft.fftfreq(Nt, d=solver.dt)
        bp = np.exp(-0.5 * ((freqs_e - f0_fit) / 8.0e12) ** 2) * (freqs_e > 0)
        analytic_e = 2.0 * np.fft.ifft(fft_e * bp)
        env_p = np.abs(analytic_e) ** 2
        inst_freq = np.gradient(np.unwrap(np.angle(analytic_e)), solver.dt) / (
            2.0 * np.pi
        )
        dlnp_dt = np.gradient(np.log(np.maximum(env_p, 1.0e-30)), solver.dt)
        win = (t_arr > 1.05e-12) & (t_arr < 1.75e-12)
        t_w = t_arr[win]
        dnu_w = inst_freq[win] - np.mean(inst_freq[win])
        dlnp_w = dlnp_dt[win] - np.mean(dlnp_dt[win])
        slope = float(np.dot(dlnp_w, dnu_w) / np.dot(dlnp_w, dlnp_w))
        return t_w, dnu_w, dlnp_w, slope

    _, dnu_0, _, _ = run_modulated_3d_fit(0.0)
    _, dnu_2, dlnp_2, slope_2 = run_modulated_3d_fit(2.0)
    t_fit_w, dnu_35, dlnp_35, slope_35 = run_modulated_3d_fit(3.5)
    slope_th_2 = float(2.0 / (4.0 * np.pi))
    slope_th_35 = float(3.5 / (4.0 * np.pi))
    rel_err_slope_2 = float(abs(slope_2 - slope_th_2) / slope_th_2)
    rel_err_slope_35 = float(abs(slope_35 - slope_th_35) / slope_th_35)
    fit_chirp_rms_alpha_0_GHz = float(np.sqrt(np.mean(dnu_0**2)) * 1e-9)

    return {
        "f_mod_GHz": f_mod_GHz,
        "bias_factors": bias_factors,
        "h_curves_dB": h_curves_dB,
        "f_r_theory_GHz": f_r_theory_GHz,
        "f_r_num_GHz": f_r_num_GHz,
        "ode_f_R_linearization_rel_err": float(
            np.max(
                np.abs(np.array(f_r_num_GHz) - np.array(f_r_theory_GHz))
                / np.array(f_r_theory_GHz)
            )
        ),
        "t_ns": t_ns[steady],
        "dnu_GHz": dnu_GHz[steady],
        "dnu_formula_GHz": dnu_formula_GHz[steady],
        "power_norm": tr_mod["I_out_total"][steady]
        / np.mean(tr_mod["I_out_total"][steady]),
        "ode_chirp_consistency_rel_err": ode_chirp_consistency_rel_err,
        "t_fit_ps": t_fit_w * 1e12,
        "dnu_fit_3p5_GHz": dnu_35 * 1e-9,
        "dnu_fit_kb_3p5_GHz": (slope_th_35 * dlnp_35) * 1e-9,
        "fit_chirp_slope_alpha_2p0": slope_2,
        "fit_chirp_slope_theory_alpha_2p0": slope_th_2,
        "fit_chirp_slope_rel_err_alpha_2p0": rel_err_slope_2,
        "fit_chirp_slope_alpha_3p5": slope_35,
        "fit_chirp_slope_theory_alpha_3p5": slope_th_35,
        "fit_chirp_slope_rel_err_alpha_3p5": rel_err_slope_35,
        "fit_chirp_rms_alpha_0_GHz": fit_chirp_rms_alpha_0_GHz,
    }


# ---------------------------------------------------------------------------
# 4. Closed-cavity carrier-photon conservation and carrier diffusion
# ---------------------------------------------------------------------------


def run_conservation_and_diffusion():
    """Run closed-cavity 3D FIT carrier-photon bookkeeping and 1D diffusion decay."""
    N_grid, L_box = 10, 1.0e-6
    grid = wakis.GridFIT3D(
        -L_box / 2,
        L_box / 2,
        -L_box / 2,
        L_box / 2,
        -L_box / 2,
        L_box / 2,
        N_grid,
        N_grid,
        N_grid,
        verbose=0,
    )
    f0 = 200.0e12
    omega_0 = 2.0 * np.pi * f0
    gamma = 2.0 * np.pi * 25.0e12
    N_tr = 1.0e24
    g_prime = 5.0e-20

    def run_cavity(alpha_H_val):
        solver = wakis.SolverFIT3D(
            grid, cfln=0.5, bc_low=["pec"] * 3, bc_high=["pec"] * 3, verbose=0
        )
        mat = SemiconductorGain(
            omega_0=omega_0,
            gamma=gamma,
            N_tr=N_tr,
            differential_gain=g_prime,
            alpha_H=alpha_H_val,
            J=0.0,
            A=0.0,
            B=0.0,
            C=0.0,
            eps_inf=2.0,
            initial_carriers=3.0 * N_tr,
            enable_stimulated=True,
        )
        solver.add_gain_material(
            mat, center=(0.0, 0.0, 0.0), size=(0.6e-6, 0.6e-6, 0.6e-6)
        )
        src = GaussianSource(frequency=f0, fwidth=0.25 * f0)
        dip = CurrentSource(
            src,
            component="z",
            center=(0.1e-6, -0.1e-6, 0.0),
            size=(0.0, 0.0, 0.0),
            amplitude=1.0e-7,
        )
        itA = grid.itA.toarray()
        iA = grid.iA.toarray()
        tA = np.divide(1.0, itA, out=np.zeros_like(itA), where=itA != 0)
        A_face = np.divide(1.0, iA, out=np.zeros_like(iA), where=iA != 0)
        dV_E = grid.L.toarray() * tA
        dV_H = grid.tL.toarray() * A_face
        ieps = solver.ieps.toarray()
        eps_r = np.divide(
            1.0, epsilon_0 * ieps, out=np.zeros_like(ieps), where=ieps != 0
        )
        g_grp = solver._gain_groups[0]
        eps_r[g_grp["idx"]] = g_grp["den"]

        Nt_src = int(np.ceil(src.end_time / solver.dt)) + 2
        for n in range(Nt_src):
            dip.update(solver, n * solver.dt)
            solver.one_step()
        solver.J.array[:] = 0.0

        bk0 = solver.get_gain_bookkeeping()
        E_old = solver.E.toarray().copy()
        H_old = solver.H.toarray().copy()
        p_prev = g_grp["poles"][0]["p"].copy()
        n_steps = 80
        t_fs = np.arange(n_steps) * solver.dt * 1e15
        carriers_rem = np.empty(n_steps)
        photons_gen = np.empty(n_steps)
        w_em_list = np.empty(n_steps)
        pol_work_step = np.empty(n_steps)
        dchi_work_step = np.empty(n_steps)

        for k in range(n_steps):
            dchi_edge_n = mat.delta_chi_bg(N=g_grp["N"])[g_grp["inv"]]
            solver.one_step()
            E_cur = solver.E.toarray().copy()
            H_cur = solver.H.toarray().copy()
            w_em_list[k] = 0.5 * epsilon_0 * np.sum(
                eps_r * dV_E * E_old**2
            ) + 0.5 * mu_0 * np.sum(dV_H * H_old * H_cur)
            p_cur = g_grp["poles"][0]["p"].copy()
            dp = p_cur - p_prev
            e_cur_g = E_cur[g_grp["idx"]]
            e_old_g = E_old[g_grp["idx"]]
            e_half = 0.5 * (e_cur_g + e_old_g)
            pol_work_step[k] = -epsilon_0 * np.sum(
                g_grp["w_dV_edge"] * (dp / solver.dt) * e_half
            )
            dchi_work_step[k] = (
                -0.5
                * (epsilon_0 / solver.dt)
                * np.sum(g_grp["w_dV_edge"] * dchi_edge_n * (e_cur_g**2 - e_old_g**2))
            )
            p_prev = p_cur
            E_old, H_old = E_cur, H_cur
            bk = solver.get_gain_bookkeeping()
            carriers_rem[k] = bk["carriers_removed"] - bk0["carriers_removed"]
            photons_gen[k] = bk["photons_generated"] - bk0["photons_generated"]

        bk_end = solver.get_gain_bookkeeping()
        return (
            t_fs,
            carriers_rem,
            photons_gen,
            w_em_list,
            pol_work_step,
            dchi_work_step,
            bk0,
            bk_end,
            solver.dt,
            g_grp["total_vol"],
        )

    (
        t_fs,
        carriers_rem,
        photons_gen,
        w_em_0,
        p_work_0,
        _,
        _,
        _,
        dt_0,
        _,
    ) = run_cavity(0.0)

    # Internal consistency check: carriers_removed vs photons_generated (both from stim_edge_rate)
    rel_err_bookkeeping_consistency = float(
        np.max(np.abs(carriers_rem - photons_gen)) / np.max(np.abs(photons_gen))
    )
    dw_dt_0 = np.diff(w_em_0) / dt_0
    rel_err_poynting_alpha_0 = float(
        np.linalg.norm(dw_dt_0 - p_work_0[:-1]) / np.linalg.norm(p_work_0[:-1])
    )

    # Closed PEC cavity discrete Poynting work balance and parametric work bound for alpha_H = 3.5 != 0
    (
        _,
        _,
        _,
        w_em_a,
        p_work_a,
        dchi_work_a,
        bka_0,
        bka_end,
        dt_a,
        vol_act_a,
    ) = run_cavity(3.5)
    dw_dt_a = np.diff(w_em_a) / dt_a
    rel_err_poynting_alpha_3p5 = float(
        np.linalg.norm(dw_dt_a - (p_work_a[:-1] + dchi_work_a[:-1]))
        / np.linalg.norm(p_work_a[:-1])
    )
    rel_err_poynting_uncorrected_alpha_3p5 = float(
        np.linalg.norm(dw_dt_a - p_work_a[:-1]) / np.linalg.norm(p_work_a[:-1])
    )
    w_stim_a = bka_end["em_work_joules"] - bka_0["em_work_joules"]
    w_dchi_a = bka_end["dchi_work_joules"] - bka_0["dchi_work_joules"]
    w_param_a = bka_end["dchi_parametric_joules"] - bka_0["dchi_parametric_joules"]
    w_stored_a = bka_end["dchi_stored_joules"] - bka_0["dchi_stored_joules"]
    u_em_max = float(np.max(w_em_a)) / vol_act_a
    dchi_parametric_bound_joules = float(
        abs(w_stim_a)
        * 3.5
        * (c * np.sqrt(2.0) * g_prime / omega_0)
        * (u_em_max / (hbar * omega_0))
    )

    # 1D carrier diffusion mode decay exp(-D k^2 t)
    D_diff = 1.5e-3  # [m^2/s]
    semi_diff = SemiconductorGain(
        omega_0=omega_0,
        gamma=gamma,
        N_tr=N_tr,
        A=0.0,
        B=0.0,
        C=0.0,
        diffusion_coeff=D_diff,
    )
    M_pts = 64
    L_diff = 6.4e-6
    dx_diff = L_diff / M_pts
    x_um = (np.arange(M_pts) + 0.5) * dx_diff * 1e6
    k_mode = 2.0 * np.pi / L_diff
    n0_prof = N_tr + 0.4 * N_tr * np.cos(k_mode * (x_um * 1e-6))
    t_end = 0.3e-9
    _, n_hist_d = semi_diff.step_rate_equations(
        Nt=200, dt=t_end / 200, N_init=n0_prof, dx=dx_diff, periodic=True
    )
    k_disc2 = (2.0 / dx_diff * np.sin(0.5 * k_mode * dx_diff)) ** 2
    n_exact_end = N_tr + 0.4 * N_tr * np.exp(-D_diff * k_disc2 * t_end) * np.cos(
        k_mode * (x_um * 1e-6)
    )
    rel_err_diff = float(
        np.linalg.norm(n_hist_d[-1] - n_exact_end)
        / np.linalg.norm(n_exact_end - N_tr)
    )
    rel_err_sum = float(
        np.max(np.abs(n_hist_d.sum(axis=1) - n0_prof.sum())) / n0_prof.sum()
    )

    return {
        "t_fs": t_fs,
        "carriers_removed": carriers_rem,
        "photons_generated": photons_gen,
        "rel_err_bookkeeping_consistency": rel_err_bookkeeping_consistency,
        "rel_err_poynting_alpha_0": rel_err_poynting_alpha_0,
        "rel_err_poynting_alpha_3p5": rel_err_poynting_alpha_3p5,
        "rel_err_poynting_uncorrected_alpha_3p5": rel_err_poynting_uncorrected_alpha_3p5,
        "dchi_work_joules_alpha_3p5": float(w_dchi_a),
        "dchi_parametric_joules_alpha_3p5": float(w_param_a),
        "dchi_stored_joules_alpha_3p5": float(w_stored_a),
        "dchi_parametric_bound_joules_alpha_3p5": dchi_parametric_bound_joules,
        "x_um": x_um,
        "n0_prof": n0_prof,
        "n_end_num": n_hist_d[-1],
        "n_end_exact": n_exact_end,
        "rel_err_diff": rel_err_diff,
        "rel_err_sum": rel_err_sum,
    }


# ---------------------------------------------------------------------------
# 5. Publication figures (wakis.figstyle)
# ---------------------------------------------------------------------------


def make_figures(spec_data, li_data, mod_data, cons_data, img_dir):
    """Create and save the two 3-panel publication figures using `wakis.figstyle`."""
    os.makedirs(img_dir, exist_ok=True)
    with apply_style():
        # --- Figure 1: Gain spectrum, Henry alpha_H, and carrier-photon conservation ---
        fig1, axes1 = new_figure(width_in=7.0, height_in=8.2, nrows=3, ncols=1)
        f_THz = spec_data["f_THz"]
        cases = spec_data["cases"]

        # Panel (a): Small-signal gain spectrum g(f, N)
        ax = axes1[0]
        spectrum_grid(ax)
        panel_label(ax, "(a)", x=-0.10, y=1.12)
        series_cfg = [
            ("abs_0.25Ntr", "Abs. $0.25\\,N_{\\mathrm{tr}}$", PALETTE[1], "o"),
            ("trans_1.0Ntr", "Transp. $1.0\\,N_{\\mathrm{tr}}$", PALETTE[7], "s"),
            ("gain_2.5Ntr", "Linear $2.5\\,N_{\\mathrm{tr}}$", PALETTE[0], "^"),
            ("log_2.5Ntr", "Log. $2.5\\,N_{\\mathrm{tr}}$", PALETTE[2], "D"),
        ]
        for key, lbl, col, mk in series_cfg:
            c_item = cases[key]
            ax.plot(
                f_THz,
                c_item["g_an"] * 1e-4,
                color=col,
                linestyle="-",
                marker="none",
                linewidth=1.8,
                label=f"{lbl} (analytic)",
            )
            ax.plot(
                f_THz[::3],
                c_item["g_fit"][::3] * 1e-4,
                color=col,
                linestyle="none",
                marker=mk,
                markersize=5.0,
                label=f"{lbl} (3D FIT)",
            )
        ax.set_ylim(-8.0, 19.5)
        ax.set_xlabel("Frequency $f$ [THz]")
        ax.set_ylabel("Gain $g(f, N)$ [$10^4\\,\\mathrm{m}^{-1}$]")
        ax.set_title("Small-Signal Semiconductor Gain Spectrum: 3D FIT Slab vs Analytic")
        ax.legend(ncol=2, loc="upper right", fontsize=8.5)

        # Panel (b): Henry alpha_H phase shift vs log amplitude gain
        ax = axes1[1]
        spectrum_grid(ax)
        panel_label(ax, "(b)", x=-0.10, y=1.12)
        dln_grid = np.linspace(0.0, 0.0075, 100)
        for k_a, col, mk in [
            ("alpha_2.0", PALETTE[0], "o"),
            ("alpha_4.0", PALETTE[1], "s"),
            ("alpha_5.0", PALETTE[2], "^"),
        ]:
            item = spec_data["alpha"][k_a]
            a_val = item["alpha_input"]
            a_meas = item["alpha_measured"]
            ax.plot(
                dln_grid * 1e3,
                a_val * dln_grid * 1e3,
                color=col,
                linestyle="-",
                marker="none",
                linewidth=1.8,
                label=f"Theory $\\alpha_H = {a_val:.1f}$",
            )
            ax.plot(
                np.array(item["dln_amp"]) * 1e3,
                np.array(item["dphi_rad"]) * 1e3,
                color=col,
                linestyle="none",
                marker=mk,
                markersize=5.8,
                label=f"3D FIT ($\\alpha_{{H,\\mathrm{{meas}}}} = {a_meas:.3f}$)",
            )
        ax.set_xlim(0.0, 7.5)
        ax.set_ylim(0.0, 46.0)
        ax.set_xlabel("Log amplitude gain $\\Delta \\ln |\\tilde{t}(\\omega_0)|$ [$10^{-3}$]")
        ax.set_ylabel("Phase shift $\\Delta \\phi(\\omega_0)$ [mrad]")
        ax.set_title("Linewidth Enhancement Factor $\\alpha_H = \\Delta \\phi / \\Delta \\ln |\\tilde{t}|$")
        ax.legend(ncol=2, loc="upper left", fontsize=8.5)

        # Panel (c): Carrier-photon conservation
        ax = axes1[2]
        spectrum_grid(ax)
        panel_label(ax, "(c)", x=-0.10, y=1.12)
        ax.plot(
            cons_data["t_fs"],
            cons_data["photons_generated"] * 1e-3,
            color=PALETTE[0],
            linestyle="-",
            marker="none",
            linewidth=2.0,
            label="Photons generated $\\Delta W_{\\mathrm{EM}} / (\\hbar \\omega_0)$ [$10^3$]",
        )
        ax.plot(
            cons_data["t_fs"][::5],
            cons_data["carriers_removed"][::5] * 1e-3,
            color=PALETTE[1],
            linestyle="none",
            marker="o",
            markersize=5.2,
            label="Carriers removed $-\\int \\Delta N\\,dV$ [$10^3$] (3D FIT)",
        )
        ax.set_xlabel("Time after seed pulse $t$ [fs]")
        ax.set_ylabel("Cumulative count [$10^3$]")
        ax.set_title(
            f"Closed-Cavity Carrier–Photon Conservation (Poynting err. = {cons_data['rel_err_poynting_alpha_0']:.2e})"
        )
        ax.legend(loc="lower right", fontsize=8.8)

        fig1_path = str(
            save(
                fig1,
                os.path.join(img_dir, "semiconductor_gain_spectrum_and_alpha.png"),
                close=True,
                auto_legend_below=False,
            )
        )

        # --- Figure 2: 1D DBR Fabry-Perot L-I curve, modulation response, and chirp ---
        fig2, axes2 = new_figure(width_in=7.0, height_in=8.2, nrows=3, ncols=1)

        # Panel (a): Steady-state L-I curve and carrier clamping
        ax = axes2[0]
        spectrum_grid(ax)
        panel_label(ax, "(a)", x=-0.10, y=1.12)
        j_norm_fine = np.linspace(0.0, 3.2, 200)
        j_fine = li_data["J_tr"] + j_norm_fine * (
            li_data["J_th"] - li_data["J_tr"]
        )
        i_an_fine = np.where(
            j_norm_fine > 1.0,
            li_data["I_an_1"][-1] * (j_norm_fine - 1.0) / (li_data["r_excess"][-1] - 1.0),
            0.0,
        )
        ax.plot(
            j_fine * 1e-9,
            i_an_fine * 1e-6,
            color=PALETTE[0],
            linestyle="-",
            marker="none",
            linewidth=2.0,
            label=f"Rate-eq. L–I ($\\eta_d = {li_data['eta_d_expected']:.2f}$)",
        )
        ax.plot(
            li_data["J_sweep"] * 1e-9,
            li_data["I_fit_1"] * 1e-6,
            color=PALETTE[1],
            linestyle="none",
            marker="o",
            markersize=6.0,
            label=f"3D FIT L–I ($\\eta_{{d,\\mathrm{{FIT}}}} = {li_data['eta_d_fit']:.2f}$)",
        )
        ax.axvline(
            li_data["J_th"] * 1e-9,
            color=PALETTE[2],
            linestyle=":",
            marker="none",
            linewidth=1.6,
            label="$J_{\\mathrm{th}}$",
        )
        ax.set_xlabel("Injection current density $J$ [$\\mathrm{GA/m^2}$]")
        ax.set_ylabel("Output $I_{\\mathrm{out},1}$ [$\\mathrm{MW/m^2}$]")
        ax.set_title("1D DBR Fabry–Pérot Diode Laser L–I Curve: 3D FIT vs Rate Equations")
        ax.legend(loc="upper left", fontsize=8.8)

        # Panel (b): Small-signal modulation response |H(f_m)|^2
        ax = axes2[1]
        spectrum_grid(ax)
        panel_label(ax, "(b)", x=-0.10, y=1.12)
        f_m = mod_data["f_mod_GHz"]
        for idx_b, bf in enumerate(mod_data["bias_factors"]):
            col = PALETTE[idx_b % len(PALETTE)]
            fr_val = mod_data["f_r_theory_GHz"][idx_b]
            ax.plot(
                f_m,
                mod_data["h_curves_dB"][f"{bf:.1f}J_th"],
                color=col,
                linestyle=("-", "--", "-.", ":")[idx_b],
                marker="none",
                linewidth=1.9,
                label=f"$J = {bf:.1f}\\,J_{{\\mathrm{{th}}}}$ ($f_R = {fr_val:.2f}$ GHz)",
            )
        ax.set_xlabel("Modulation frequency $f_m$ [GHz]")
        ax.set_ylabel("Response $|H(f_m)|^2$ [dB]")
        ax.set_ylim(-22.0, 26.0)
        ax.set_title("Small-Signal Modulation Response & Relaxation-Oscillation Resonance")
        ax.legend(ncol=2, loc="upper right", fontsize=8.5)

        # Panel (c): 3D FIT demodulated frequency chirp vs Koch-Bowers (alpha_H / 4 pi) d(ln P)/dt
        ax = axes2[2]
        spectrum_grid(ax)
        panel_label(ax, "(c)", x=-0.10, y=1.12)
        t_ps = mod_data["t_fit_ps"]
        ax.plot(
            t_ps,
            mod_data["dnu_fit_3p5_GHz"],
            color=PALETTE[0],
            linestyle="-",
            marker="none",
            linewidth=2.0,
            label="3D FIT demodulated $\\Delta \\nu_{\\mathrm{FIT}}(t)$",
        )
        ax.plot(
            t_ps[::15],
            mod_data["dnu_fit_kb_3p5_GHz"][::15],
            color=PALETTE[1],
            linestyle="none",
            marker="s",
            markersize=4.8,
            label="Koch–Bowers $(\\alpha_H / 4\\pi)\\,d\\ln P_{\\mathrm{FIT}} / dt$",
        )
        ax.set_xlabel("Time $t$ [ps]")
        ax.set_ylabel("Chirp $\\Delta \\nu(t)$ [GHz]")
        ax.set_title(
            f"3D FIT Koch–Bowers Chirp Slope ($\\alpha_H = 3.5$, slope err. = {mod_data['fit_chirp_slope_rel_err_alpha_3p5']:.2e})"
        )
        ax.legend(ncol=2, loc="upper right", fontsize=8.5)

        fig2_path = str(
            save(
                fig2,
                os.path.join(img_dir, "semiconductor_fp_laser_li_and_chirp.png"),
                close=True,
                auto_legend_below=False,
            )
        )

    return fig1_path, fig2_path


def main():
    t0 = time.time()
    print("=== Example 015: Semiconductor Gain Medium & 1D DBR Fabry-Perot Diode Laser ===")

    spec_data = run_slab_gain_and_alpha()
    print(
        f"[1/4] Slab gain spectrum & alpha_H completed: "
        f"max |t|^2 rel_err = {spec_data['cases']['gain_2.5Ntr']['rel_err_t2']:.3e}, "
        f"alpha_H(4.0) meas = {spec_data['alpha']['alpha_4.0']['alpha_measured']:.4f}"
    )

    li_data = run_fp_diode_laser_li()
    print(
        f"[2/4] 1D DBR FP diode laser L-I completed: "
        f"eta_d_fit = {li_data['eta_d_fit']:.4f} (theory {li_data['eta_d_expected']:.4f}), "
        f"tau_p_cold = {li_data['tau_p_cold_fs']:.3f} fs (harminv Q={li_data['Q_cold']:.2f}), "
        f"f_R_fit = {li_data['f_R_fit_THz']:.3f} THz (cold-ringdown DBR {li_data['f_R_an_THz']:.3f} THz, "
        f"analytic L_pen {li_data['f_R_analytic_lpen_THz']:.3f} THz, bare {li_data['f_R_bare_THz']:.3f} THz)"
    )

    mod_data = run_modulation_and_chirp()
    print(
        f"[3/4] Modulation & 3D FIT chirp completed: "
        f"3D FIT chirp slope(3.5) rel_err = {mod_data['fit_chirp_slope_rel_err_alpha_3p5']:.3e}, "
        f"ODE linearization f_R consistency = {mod_data['ode_f_R_linearization_rel_err']:.3e}"
    )

    cons_data = run_conservation_and_diffusion()
    print(
        f"[4/4] Conservation & diffusion completed: "
        f"Poynting(alpha_H=3.5) rel_err = {cons_data['rel_err_poynting_alpha_3p5']:.3e}, "
        f"diffusion rel_err = {cons_data['rel_err_diff']:.3e}"
    )

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    img_dir = os.path.join(repo_root, "docs", "img")
    f1, f2 = make_figures(spec_data, li_data, mod_data, cons_data, img_dir)
    elapsed = time.time() - t0

    # Write every quoted number to JSON with repo-relative paths
    json_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(json_dir, exist_ok=True)
    json_path = os.path.join(json_dir, "015_semiconductor_fp_laser.json")
    summary_payload = {
        "example": "015_semiconductor_fp_laser.py",
        "elapsed_seconds": float(elapsed),
        "figures": [
            os.path.relpath(f1, repo_root),
            os.path.relpath(f2, repo_root),
        ],
        "slab_gain_spectrum": {
            k: {
                "g_center_fit_m_inv": v["g_center_fit"],
                "g_center_an_m_inv": v["g_center_an"],
                "rel_err_t2": v["rel_err_t2"],
                "rel_err_center": v["rel_err_center"],
            }
            for k, v in spec_data["cases"].items()
        },
        "henry_alpha_factor": spec_data["alpha"],
        "fabry_perot_li": {
            "J_tr_A_per_m2": li_data["J_tr"],
            "J_th_theory_A_per_m2": li_data["J_th"],
            "J_th_fit_A_per_m2": li_data["J_th_fit"],
            "J_th_rel_err": li_data["J_th_rel_err"],
            "N_tr_m_inv3": li_data["N_tr"],
            "N_th_m_inv3": li_data["N_th"],
            "eta_d_theory": li_data["eta_d_expected"],
            "eta_d_fit": li_data["eta_d_fit"],
            "eta_d_rel_err": li_data["eta_d_rel_err"],
            "max_rel_err_N_clamped": li_data["max_rel_err_N_clamped"],
            "max_rel_err_I_above": li_data["max_rel_err_I_above"],
            "f_cold_THz": li_data["f_cold_THz"],
            "Q_cold": li_data["Q_cold"],
            "tau_p_cold_fs": li_data["tau_p_cold_fs"],
            "L_eff_dbr_m": li_data["L_eff_dbr_m"],
            "Gamma_z_dbr": li_data["Gamma_z_dbr"],
            "L_pen_analytic_m": li_data["L_pen_analytic_m"],
            "L_eff_analytic_dbr_m": li_data["L_eff_analytic_dbr_m"],
            "Gamma_z_analytic_dbr": li_data["Gamma_z_analytic_dbr"],
            "f_R_fit_THz": li_data["f_R_fit_THz"],
            "f_R_bare_THz": li_data["f_R_bare_THz"],
            "f_R_bare_rel_err": li_data["f_R_bare_rel_err"],
            "f_R_theory_THz": li_data["f_R_an_THz"],
            "f_R_fit_rel_err": li_data["f_R_fit_rel_err"],
            "f_R_analytic_lpen_THz": li_data["f_R_analytic_lpen_THz"],
            "f_R_analytic_lpen_rel_err": li_data["f_R_analytic_lpen_rel_err"],
            "r_excess": li_data["r_excess"].tolist(),
            "I_fit_1_W_per_m2": li_data["I_fit_1"].tolist(),
            "I_an_1_W_per_m2": li_data["I_an_1"].tolist(),
            "N_clamped_m_inv3": li_data["N_clamped"].tolist(),
        },
        "modulation_and_chirp": {
            "bias_factors": mod_data["bias_factors"],
            "f_R_theory_GHz": mod_data["f_r_theory_GHz"],
            "f_R_numerical_GHz": mod_data["f_r_num_GHz"],
            "ode_f_R_linearization_rel_err": mod_data["ode_f_R_linearization_rel_err"],
            "ode_chirp_consistency_rel_err": mod_data["ode_chirp_consistency_rel_err"],
            "fit_chirp_slope_alpha_2p0": mod_data["fit_chirp_slope_alpha_2p0"],
            "fit_chirp_slope_theory_alpha_2p0": mod_data[
                "fit_chirp_slope_theory_alpha_2p0"
            ],
            "fit_chirp_slope_rel_err_alpha_2p0": mod_data[
                "fit_chirp_slope_rel_err_alpha_2p0"
            ],
            "fit_chirp_slope_alpha_3p5": mod_data["fit_chirp_slope_alpha_3p5"],
            "fit_chirp_slope_theory_alpha_3p5": mod_data[
                "fit_chirp_slope_theory_alpha_3p5"
            ],
            "fit_chirp_slope_rel_err_alpha_3p5": mod_data[
                "fit_chirp_slope_rel_err_alpha_3p5"
            ],
            "fit_chirp_rms_alpha_0_GHz": mod_data["fit_chirp_rms_alpha_0_GHz"],
        },
        "conservation_and_diffusion": {
            "rel_err_bookkeeping_consistency": cons_data[
                "rel_err_bookkeeping_consistency"
            ],
            "rel_err_poynting_alpha_0": cons_data["rel_err_poynting_alpha_0"],
            "rel_err_poynting_alpha_3p5": cons_data["rel_err_poynting_alpha_3p5"],
            "rel_err_poynting_uncorrected_alpha_3p5": cons_data[
                "rel_err_poynting_uncorrected_alpha_3p5"
            ],
            "dchi_work_joules_alpha_3p5": cons_data["dchi_work_joules_alpha_3p5"],
            "dchi_parametric_joules_alpha_3p5": cons_data[
                "dchi_parametric_joules_alpha_3p5"
            ],
            "dchi_stored_joules_alpha_3p5": cons_data["dchi_stored_joules_alpha_3p5"],
            "dchi_parametric_bound_joules_alpha_3p5": cons_data[
                "dchi_parametric_bound_joules_alpha_3p5"
            ],
            "rel_err_diffusion_mode": cons_data["rel_err_diff"],
            "rel_err_diffusion_sum": cons_data["rel_err_sum"],
            "final_photons_generated": float(cons_data["photons_generated"][-1]),
            "final_carriers_removed": float(cons_data["carriers_removed"][-1]),
        },
    }
    with open(json_path, "w", encoding="utf-8") as fp:
        json.dump(summary_payload, fp, indent=2)

    print(f"Saved figures:\n  {f1}\n  {f2}")
    print(f"Saved benchmark JSON:\n  {json_path}")
    print(f"Completed in {elapsed:.1f} s.")


if __name__ == "__main__":
    main()
