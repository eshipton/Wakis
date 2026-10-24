# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Example 012 — 1D DBR Fabry–Pérot Laser with a Four-Level Atomic Gain Medium
===========================================================================

Demonstrates time-domain simulation of a Fabry–Pérot microcavity laser in
`SolverFIT3D` using `FourLevelGain` (`wakis.gain`), following the semiclassical
Maxwell–Bloch / rate-equation formalism of S.-H. Chang and A. Taflove,
*Opt. Express* **12**(16), 3827–3833 (2004), doi:10.1364/OPEX.12.003827.

Physical setup
--------------
* **Cavity geometry**: A 1D active cavity of length ``L_cav = 1.2 um`` (two
  half-wavelengths at ``lambda_0 = 1.2 um``, ``f_0 = 249.83 THz``) bounded on
  both sides by symmetric 3-layer quarter-wave distributed Bragg reflectors
  ``(H L H)`` with ``n_H = 2.0`` (``d_H = 150 nm = 6 dz``) and ``n_L = 1.0``
  (``d_L = 300 nm = 12 dz``).
* **Mirror reflectance and threshold**: At the Bragg wavelength ``lambda_0``,
  each ``(H L H)`` mirror has analytical power reflectance
  ``R = ((n_H^4 - 1) / (n_H^4 + 1))^2 = (15 / 17)^2 = 0.77855``, yielding a
  cold-cavity threshold gain coefficient
  ``g_th = ln(1 / R) / L_cav = 2.0861e5 m^-1``.
* **Four-level gain medium**: Total atomic density ``N_total = 5.0e23 m^-3``,
  lasing transition ``1 <-> 2`` at ``omega_a = 2 * pi * f_0`` with homogeneous
  linewidth ``gamma_a / (2 * pi) = 40 THz``, upper-state lifetime
  ``tau_21 = 150 fs``, and fast nonradiative relaxation
  ``tau_32 = tau_10 = 5 fs`` (``tau_21 >> tau_10``).
* **Seeding**: Because semiclassical Maxwell–rate equations do not include
  stochastic spontaneous-emission Langevin noise, a weak broadband Gaussian
  current sheet pulse ``J_x(t)`` inside the cavity seeds the cavity mode. Below
  threshold (``W_p < W_{p,th}``) the seed decays away; above threshold
  (``W_p > W_{p,th}``) it amplifies through damped relaxation oscillations into
  steady-state single-mode continuous-wave oscillation at ``f_cav`` with
  clamped modal inversion ``Delta N = Delta N_th``.

Saves two publication-style figures in ``docs/img/``:
  1. ``docs/img/fabry_perot_laser_schematic_and_li.png``
  2. ``docs/img/fabry_perot_laser_dynamics_and_spectrum.png``
"""

import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c, epsilon_0

import wakis
from wakis.gain import (
    FourLevelGain,
    dbr_reflectance,
    fabry_perot_laser_output,
    fabry_perot_threshold_gain,
)
from wakis.sources import CurrentSource, GaussianSource

# High-contrast publication plot styling (Okabe-Ito color-blind-safe palette)
plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": "sans-serif",
        "font.size": 9.5,
        "axes.labelsize": 10.0,
        "axes.titlesize": 10.5,
        "xtick.labelsize": 9.0,
        "ytick.labelsize": 9.0,
        "legend.fontsize": 8.5,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "lines.linewidth": 1.8,
    }
)

PALETTE = {
    "blue": "#0072B2",
    "orange": "#D55E00",
    "green": "#009E73",
    "purple": "#CC79A7",
    "yellow": "#E69F00",
    "sky": "#56B4E9",
    "black": "#222222",
    "grey": "#666666",
}


def setup_parameters():
    """Return physical and numerical parameters of the 1D DBR Fabry-Perot laser."""
    lam0 = 1.2e-6  # Bragg & atomic transition wavelength [m]
    f0 = c / lam0  # 249.827 THz
    omega_a = 2.0 * np.pi * f0
    gamma_a = 2.0 * np.pi * 40.0e12  # homogeneous linewidth [rad/s]
    dz = 25.0e-9  # 48 cells/lambda in vacuum, 24 cells/lambda in n_H = 2.0

    n_H, n_L = 2.0, 1.0
    d_H = lam0 / (4.0 * n_H)  # 150 nm = 6 * dz
    d_L = lam0 / (4.0 * n_L)  # 300 nm = 12 * dz
    n_pairs = 1  # (H L H) mirror on each side
    R_mirror = dbr_reflectance(n_H, n_L, n_pairs, n_in=1.0, n_out=1.0, extra_high=True)

    L_cav = 1.2e-6  # 2 * (lam0 / 2) = 48 * dz
    g_th = fabry_perot_threshold_gain(L_cav, R_mirror, R_mirror)

    tau_21 = 150.0e-15
    tau_10 = 5.0e-15
    tau_32 = 5.0e-15
    N_total = 5.0e23

    fg_ref = FourLevelGain(
        omega_a=omega_a,
        gamma_a=gamma_a,
        tau_21=tau_21,
        tau_10=tau_10,
        tau_32=tau_32,
        N_total=N_total,
        pump_rate=0.0,
        eps_inf=1.0,
    )
    W_p_th = fg_ref.threshold_pump_rate(g_th)
    dn_th = g_th / fg_ref.emission_cross_section
    I_sat = fg_ref.saturation_intensity(W_p=W_p_th)

    return {
        "lam0": lam0,
        "f0": f0,
        "omega_a": omega_a,
        "gamma_a": gamma_a,
        "dz": dz,
        "n_H": n_H,
        "n_L": n_L,
        "d_H": d_H,
        "d_L": d_L,
        "n_pairs": n_pairs,
        "R_mirror": R_mirror,
        "L_cav": L_cav,
        "g_th": g_th,
        "tau_21": tau_21,
        "tau_10": tau_10,
        "tau_32": tau_32,
        "N_total": N_total,
        "fg_ref": fg_ref,
        "W_p_th": W_p_th,
        "dn_th": dn_th,
        "I_sat": I_sat,
        "z_min": -1.2e-6,
        "z_max": 2.4e-6,
        "n_pml": 10,
    }


def build_cavity_solver(p):
    """Create a 1D-periodic SolverFIT3D with static quarter-wave DBR mirrors."""
    Nz = int(round((p["z_max"] - p["z_min"]) / p["dz"]))
    grid = wakis.GridFIT3D(
        -1.0e-6,
        1.0e-6,
        -1.0e-6,
        1.0e-6,
        p["z_min"],
        p["z_max"],
        2,
        2,
        Nz,
        verbose=0,
    )
    solver = wakis.SolverFIT3D(
        grid,
        cfln=0.8,
        bc_low=["periodic", "periodic", "cpml"],
        bc_high=["periodic", "periodic", "cpml"],
        n_pml=p["n_pml"],
        verbose=0,
    )
    ieps_arr = solver.ieps.array
    for m in range(p["n_pairs"] + 1):
        z_left = -(m * (p["d_H"] + p["d_L"]) + 0.5 * p["d_H"])
        z_right = p["L_cav"] + m * (p["d_H"] + p["d_L"]) + 0.5 * p["d_H"]
        for z_c in (z_left, z_right):
            idx_e, w_e = solver._box_edges(
                (0.0, 0.0, z_c), (4.0e-6, 4.0e-6, p["d_H"]), dual=False
            )
            eps_s = 1.0 / (epsilon_0 * ieps_arr[idx_e])
            eps_new = (1.0 - w_e) * eps_s + w_e * (p["n_H"] ** 2)
            ieps_arr[idx_e] = 1.0 / (epsilon_0 * eps_new)
    solver.update_tensors("ieps")
    return solver


def run_cold_cavity(p):
    """Run a cold-cavity ringdown simulation to extract mode profile, f_cav, and Q."""
    solver = build_cavity_solver(p)
    src = GaussianSource(frequency=p["f0"], fwidth=0.05 * p["f0"])
    sheet = CurrentSource(
        src,
        component="x",
        center=(0.0, 0.0, 0.25 * p["L_cav"]),
        size=(2.0e-6, 2.0e-6, 0.0),
        amplitude=1.0,
    )
    k_cav = int(np.argmin(np.abs(solver.z - 0.75 * p["L_cav"])))
    idx_cav = k_cav * solver.Nx * solver.Ny
    Nt = int((src.end_time + 300.0e-15) / solver.dt)
    ex_cav = np.empty(Nt)
    mode_profile_acc = np.zeros(solver.Nz)
    n_snap = int((src.end_time + 60.0e-15) / solver.dt)
    stride_z = solver.Nx * solver.Ny

    for n in range(Nt):
        sheet.update(solver, n * solver.dt)
        solver.one_step()
        ex_cav[n] = solver.E.array[idx_cav]
        if n >= n_snap:
            ex_z = solver.E.array[0 : solver.N : stride_z]
            mode_profile_acc += ex_z**2

    t = np.arange(Nt) * solver.dt
    mask = t > src.end_time + 30.0e-15
    n_fft = 32768
    freqs = np.fft.rfftfreq(n_fft, d=solver.dt)
    spec = np.abs(np.fft.rfft(ex_cav[mask], n=n_fft))
    f_cav = freqs[np.argmax(spec)]

    win = int(round((1.0 / p["f0"]) / solver.dt))
    env2 = np.convolve(ex_cav[mask] ** 2, np.ones(win) / win, mode="valid")
    t_env = t[mask][: len(env2)]
    kappa_cold = -np.polyfit(t_env, np.log(env2), 1)[0]
    Q_cold = 2.0 * np.pi * f_cav / kappa_cold

    mode_profile = mode_profile_acc / np.max(mode_profile_acc)
    return {
        "z": solver.z.copy(),
        "mode_profile": mode_profile,
        "f_cav": f_cav,
        "Q_cold": Q_cold,
        "kappa_cold": kappa_cold,
        "freqs": freqs,
        "spec": spec / np.max(spec),
    }


def run_laser_point(p, pump_ratio, total_time=0.95e-12, spatial_averaging=False):
    """Run one active Fabry-Perot laser simulation at W_p = pump_ratio * W_{p,th}."""
    wp = pump_ratio * p["W_p_th"]
    solver = build_cavity_solver(p)
    fg = FourLevelGain(
        omega_a=p["omega_a"],
        gamma_a=p["gamma_a"],
        tau_21=p["tau_21"],
        tau_10=p["tau_10"],
        tau_32=p["tau_32"],
        N_total=p["N_total"],
        pump_rate=wp,
        eps_inf=1.0,
        initial_populations="steady_state" if wp > 0 else "ground",
        enable_stimulated=True,
        spatial_averaging=spatial_averaging,
    )
    solver.add_gain_material(
        fg,
        center=(0.0, 0.0, 0.5 * p["L_cav"]),
        size=(4.0e-6, 4.0e-6, p["L_cav"]),
    )
    src = GaussianSource(frequency=p["f0"], fwidth=0.08 * p["f0"])
    sheet = CurrentSource(
        src,
        component="x",
        center=(0.0, 0.0, 0.25 * p["L_cav"]),
        size=(2.0e-6, 2.0e-6, 0.0),
        amplitude=2.0,
    )
    z_out = p["L_cav"] + 2.0 * p["d_H"] + p["d_L"] + 0.2e-6
    k_out = int(np.argmin(np.abs(solver.z - z_out)))
    idx_out = k_out * solver.Nx * solver.Ny

    Nt = int(total_time / solver.dt)
    ex_out = np.empty(Nt)
    n1_hist = np.empty(Nt)
    n2_hist = np.empty(Nt)
    g_grp = solver._gain_groups[0]
    cw = g_grp["cell_weight"] / g_grp["eff_n_cells"]

    for n in range(Nt):
        sheet.update(solver, n * solver.dt)
        solver.one_step()
        ex_out[n] = solver.E.array[idx_out]
        N_arr = g_grp["N"]
        n1_hist[n] = float(N_arr[1] @ cw)
        n2_hist[n] = float(N_arr[2] @ cw)

    t = np.arange(Nt) * solver.dt
    n_avg = int((20.0 / p["f0"]) / solver.dt)
    I_out = c * epsilon_0 * float(np.mean(ex_out[-n_avg:] ** 2))
    return {
        "t": t,
        "dt": solver.dt,
        "ex_out": ex_out,
        "n1": n1_hist,
        "n2": n2_hist,
        "dn": n2_hist - n1_hist,
        "I_out": I_out,
    }


def make_figures(p, cold, sweep_shb, sweep_hom, traces, img_dir):
    """Generate and save the two 3-panel publication figures."""
    os.makedirs(img_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Figure 1: Setup schematic, 4-level inversion curve, and L-I curve
    # ------------------------------------------------------------------
    fig1, axes1 = plt.subplots(1, 3, figsize=(14.2, 4.3), constrained_layout=True)

    # Panel (a): Labelled 1D Fabry-Perot DBR cavity schematic + standing-wave mode
    ax = axes1[0]
    z_um = cold["z"] * 1e6
    z_min_um, z_max_um = p["z_min"] * 1e6, p["z_max"] * 1e6
    pml_w_um = p["n_pml"] * p["dz"] * 1e6
    d_H_um, d_L_um, L_cav_um = p["d_H"] * 1e6, p["d_L"] * 1e6, p["L_cav"] * 1e6

    # Shade regions: CPML, DBR H/L layers, Active Gain Cavity
    ax.axvspan(z_min_um, z_min_um + pml_w_um, color="#DDDDDD", alpha=0.8, label="CPML (10 dz)")
    ax.axvspan(z_max_um - pml_w_um, z_max_um, color="#DDDDDD", alpha=0.8)
    for m in range(p["n_pairs"] + 1):
        zl_hi = -(m * (d_H_um + d_L_um))
        zl_lo = zl_hi - d_H_um
        ax.axvspan(
            zl_lo,
            zl_hi,
            color=PALETTE["sky"],
            alpha=0.45,
            label="DBR H ($n_H=2.0$, $6\\,\\Delta z$)" if m == 0 else None,
        )
        zr_lo = L_cav_um + m * (d_H_um + d_L_um)
        zr_hi = zr_lo + d_H_um
        ax.axvspan(zr_lo, zr_hi, color=PALETTE["sky"], alpha=0.45)

    ax.axvspan(
        0.0,
        L_cav_um,
        color=PALETTE["orange"],
        alpha=0.20,
        label="Active 4-level gain ($L_{\\mathrm{cav}}=1.2\\,\\mu$m)",
    )

    # Plot refractive index profile n(z) and resonant mode |E_x(z)|^2
    n_prof = np.ones_like(z_um)
    for m in range(p["n_pairs"] + 1):
        zl_hi = -(m * (d_H_um + d_L_um))
        zl_lo = zl_hi - d_H_um
        zr_lo = L_cav_um + m * (d_H_um + d_L_um)
        zr_hi = zr_lo + d_H_um
        n_prof[(z_um >= zl_lo) & (z_um <= zl_hi)] = p["n_H"]
        n_prof[(z_um >= zr_lo) & (z_um <= zr_hi)] = p["n_H"]

    ax.step(
        z_um,
        n_prof / 2.2,
        where="mid",
        color=PALETTE["black"],
        lw=1.3,
        ls="--",
        label="Index profile $n(z) / 2.2$",
    )
    ax.plot(
        z_um,
        cold["mode_profile"],
        color=PALETTE["blue"],
        lw=1.9,
        label="Cavity mode $|E_x(z)|^2$ (norm.)",
    )
    ax.axvline(0.3, color=PALETTE["purple"], ls=":", lw=1.5, label="Seed $J_x$ ($z=0.3\\,\\mu$m)")
    ax.axvline(2.0, color=PALETTE["green"], ls="-.", lw=1.5, label="Monitor ($z=2.0\\,\\mu$m)")

    ax.set_xlim(z_min_um, z_max_um)
    ax.set_ylim(0.0, 1.18)
    ax.set_xlabel("Longitudinal coordinate $z$ [$\\mu$m] ($\\Delta z = 25$ nm)")
    ax.set_ylabel("Normalized mode $|E_x(z)|^2$ & $n(z)/2.2$ [-]")
    ax.set_title("(a) 1D DBR Fabry–Pérot Laser Setup & Mode")
    ax.text(
        0.02,
        0.93,
        "Transverse x,y: periodic | z: CPML",
        transform=ax.transAxes,
        fontsize=8.5,
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.85, "edgecolor": "#CCCCCC"},
    )
    ax.legend(loc="lower left", framealpha=0.92, ncol=1)

    # Panel (b): Steady-state unsaturated population inversion vs pump rate + 4-level diagram inset
    ax = axes1[1]
    r_grid = np.linspace(0.0, 3.2, 120)
    wp_grid = r_grid * p["W_p_th"]
    dn_unsat = np.array(
        [p["fg_ref"].steady_state_inversion(W_p=wp) for wp in wp_grid]
    ) / p["dn_th"]
    dn_clamped_hom = np.minimum(dn_unsat, 1.0)
    s_shb = np.where(
        r_grid > 1.0,
        0.25 * (4.0 * r_grid - 1.0 - np.sqrt(8.0 * r_grid + 1.0)),
        0.0,
    )
    dn_clamped_shb = np.where(
        r_grid > 1.0, dn_unsat / np.sqrt(1.0 + 2.0 * s_shb), dn_unsat
    )

    ax.plot(
        r_grid,
        dn_unsat,
        color=PALETTE["grey"],
        ls="--",
        label="Unsaturated $\\Delta N_0(W_p) / \\Delta N_{\\mathrm{th}}$",
    )
    ax.plot(
        r_grid,
        dn_clamped_shb,
        color=PALETTE["blue"],
        lw=2.0,
        label="Standing-wave $\\langle\\Delta N\\rangle_z / \\Delta N_{\\mathrm{th}}$ (SHB)",
    )
    ax.plot(
        r_grid,
        dn_clamped_hom,
        color=PALETTE["green"],
        ls="-.",
        lw=1.9,
        label="Modal / homogenized $\\Delta N / \\Delta N_{\\mathrm{th}}$",
    )
    ax.plot(
        sweep_shb["ratios"],
        sweep_shb["dn_final"] / p["dn_th"],
        "o",
        color=PALETTE["blue"],
        ms=6,
        label="3D FIT (local cells, SHB)",
    )
    ax.plot(
        sweep_hom["ratios"],
        sweep_hom["dn_final"] / p["dn_th"],
        "s",
        color=PALETTE["green"],
        ms=5.5,
        label="3D FIT (homogenized)",
    )
    ax.axvline(1.0, color=PALETTE["orange"], ls=":", lw=1.4)
    ax.annotate(
        "Threshold\n$W_p = W_{p,\\mathrm{th}}$",
        xy=(1.0, 1.0),
        xytext=(0.18, 1.55),
        arrowprops={"arrowstyle": "->", "color": PALETTE["orange"], "lw": 1.2},
        fontsize=8.5,
        color=PALETTE["orange"],
    )
    ax.text(
        0.04,
        0.74,
        "4-Level Scheme:\n"
        "$|0\\rangle \\rightarrow |3\\rangle$ ($W_p$), $|3\\rangle \\rightarrow |2\\rangle$ ($\\tau_{32}=5$ fs)\n"
        "$|2\\rangle \\rightarrow |1\\rangle$ ($\\omega_a,\\,\\tau_{21}=150$ fs), $|1\\rangle \\rightarrow |0\\rangle$ ($\\tau_{10}=5$ fs)",
        transform=ax.transAxes,
        fontsize=8.5,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "#F9F9F9", "edgecolor": "#CCCCCC"},
    )
    ax.set_xlabel("Normalized pump rate $W_p / W_{p,\\mathrm{th}}$ [-]")
    ax.set_ylabel("Normalized inversion $\\Delta N / \\Delta N_{\\mathrm{th}}$ [-]")
    ax.set_title("(b) Population Inversion & Threshold Clamping")
    ax.legend(loc="lower right", framealpha=0.92)

    # Panel (c): Laser L-I curve (Output Intensity vs Pump Rate)
    ax = axes1[2]
    I_an_shb = (
        fabry_perot_laser_output(
            r_grid, p["I_sat"], p["R_mirror"], p["R_mirror"], spatial_hole_burning=True
        )
        * 1e-6
    )
    I_an_hom = (
        fabry_perot_laser_output(
            r_grid, p["I_sat"], p["R_mirror"], p["R_mirror"], spatial_hole_burning=False
        )
        * 1e-6
    )
    ax.plot(
        r_grid,
        I_an_shb,
        color=PALETTE["blue"],
        lw=2.0,
        label="Analytic L–I (standing-wave SHB)",
    )
    ax.plot(
        r_grid,
        I_an_hom,
        color=PALETTE["green"],
        ls="-.",
        lw=1.9,
        label="Analytic L–I (homogenized $s = r - 1$)",
    )
    ax.plot(
        sweep_shb["ratios"],
        sweep_shb["I_out"] * 1e-6,
        "o",
        color=PALETTE["blue"],
        ms=6.5,
        label="3D FIT (`spatial_averaging=False`)",
    )
    ax.plot(
        sweep_hom["ratios"],
        sweep_hom["I_out"] * 1e-6,
        "s",
        color=PALETTE["green"],
        ms=6.0,
        label="3D FIT (`spatial_averaging=True`)",
    )
    ax.axvline(1.0, color=PALETTE["orange"], ls=":", lw=1.4, label="Cold-cavity threshold $r=1$")
    ax.set_xlabel("Normalized pump rate $W_p / W_{p,\\mathrm{th}}$ [-]")
    ax.set_ylabel("Steady-state output intensity $I_{\\mathrm{out}}$ [MW/m$^2$]")
    ax.set_title("(c) Fabry–Pérot Laser L–I Curve")
    ax.legend(loc="upper left", framealpha=0.92)

    fig1_path = os.path.join(img_dir, "fabry_perot_laser_schematic_and_li.png")
    fig1.savefig(fig1_path, dpi=180)
    plt.close(fig1)

    # ------------------------------------------------------------------
    # Figure 2: Relaxation oscillations, inversion dynamics, and spectrum
    # ------------------------------------------------------------------
    fig2, axes2 = plt.subplots(1, 3, figsize=(14.2, 4.3), constrained_layout=True)

    tr_above = traces["above_hom"]
    tr_shb = traces["above_shb"]
    tr_sub = traces["sub"]
    t_ps = tr_above["t"] * 1e12

    # Cycle-averaged intensity envelope (moving average over 2 optical periods)
    win = max(int(round((2.0 / p["f0"]) / tr_above["dt"])), 1)
    kernel = np.ones(win) / win
    I_env_hom = (
        c * epsilon_0 * np.convolve(tr_above["ex_out"] ** 2, kernel, mode="valid") * 1e-6
    )
    I_env_shb = (
        c * epsilon_0 * np.convolve(tr_shb["ex_out"] ** 2, kernel, mode="valid") * 1e-6
    )
    I_env_sub = (
        c * epsilon_0 * np.convolve(tr_sub["ex_out"] ** 2, kernel, mode="valid") * 1e-6
    )
    t_env_ps = t_ps[: len(I_env_hom)]

    # Panel (a): Output intensity envelope & relaxation oscillations
    ax = axes2[0]
    ax.plot(
        t_env_ps,
        I_env_hom,
        color=PALETTE["green"],
        lw=1.9,
        label="$W_p = 2.5\\,W_{p,\\mathrm{th}}$ (homogenized)",
    )
    ax.plot(
        t_env_ps,
        I_env_shb,
        color=PALETTE["blue"],
        ls="--",
        lw=1.8,
        label="$W_p = 2.5\\,W_{p,\\mathrm{th}}$ (standing-wave SHB)",
    )
    ax.plot(
        t_env_ps,
        I_env_sub * 1e4,
        color=PALETTE["grey"],
        ls=":",
        lw=1.5,
        label="$W_p = 0.5\\,W_{p,\\mathrm{th}}$ ($\\times 10^4$, decays)",
    )
    ax.axhline(
        fabry_perot_laser_output(2.5, p["I_sat"], p["R_mirror"], p["R_mirror"], False) * 1e-6,
        color=PALETTE["green"],
        ls=":",
        alpha=0.7,
    )
    ax.axhline(
        fabry_perot_laser_output(2.5, p["I_sat"], p["R_mirror"], p["R_mirror"], True) * 1e-6,
        color=PALETTE["blue"],
        ls=":",
        alpha=0.7,
    )
    ax.set_xlabel("Time $t$ [ps]")
    ax.set_ylabel("Output intensity envelope $I_{\\mathrm{out}}(t)$ [MW/m$^2$]")
    ax.set_title("(a) Laser Turn-On & Relaxation Oscillations")
    ax.legend(loc="upper right", framealpha=0.92)

    # Panel (b): Population inversion dynamics N_2(t), N_1(t), and Delta N(t)
    ax = axes2[1]
    ax.plot(
        t_ps,
        tr_above["n2"] / p["dn_th"],
        color=PALETTE["green"],
        lw=1.9,
        label="Upper level $N_2(t) / \\Delta N_{\\mathrm{th}}$ (homog.)",
    )
    ax.plot(
        t_ps,
        tr_above["dn"] / p["dn_th"],
        color=PALETTE["orange"],
        ls="--",
        lw=1.8,
        label="Inversion $\\Delta N(t) / \\Delta N_{\\mathrm{th}}$ (homog.)",
    )
    ax.plot(
        t_ps,
        tr_shb["dn"] / p["dn_th"],
        color=PALETTE["blue"],
        ls="-.",
        lw=1.7,
        label="Spatial avg. $\\langle\\Delta N(t)\\rangle_z / \\Delta N_{\\mathrm{th}}$ (SHB)",
    )
    ax.plot(
        t_ps,
        tr_above["n1"] / p["dn_th"],
        color=PALETTE["purple"],
        ls=":",
        lw=1.6,
        label="Lower level $N_1(t) / \\Delta N_{\\mathrm{th}}$ ($\\tau_{10} \\ll \\tau_{21}$)",
    )
    ax.axhline(
        1.0,
        color=PALETTE["black"],
        ls=":",
        lw=1.3,
        label="Threshold $\\Delta N_{\\mathrm{th}} = g_{\\mathrm{th}} / \\sigma_{21}$",
    )
    ax.set_xlabel("Time $t$ [ps]")
    ax.set_ylabel("Normalized population density [$\\Delta N_{\\mathrm{th}}$]")
    ax.set_title("(b) Atomic Level Populations & Inversion Clamping")
    ax.legend(loc="upper right", framealpha=0.92)

    # Panel (c): Lasing emission spectrum vs cold-cavity mode and gain curve
    ax = axes2[2]
    n_steady = len(tr_above["ex_out"]) // 2
    n_fft = 32768
    freqs_THz = np.fft.rfftfreq(n_fft, d=tr_above["dt"]) * 1e-12
    spec_lase = np.abs(np.fft.rfft(tr_above["ex_out"][n_steady:], n=n_fft)) ** 2
    spec_lase /= np.max(spec_lase)

    f_band = np.linspace(200.0, 300.0, 400)
    g_profile = p["fg_ref"].small_signal_gain(
        f=f_band * 1e12, W_p=2.5 * p["W_p_th"]
    )
    g_profile /= np.max(g_profile)

    ax.plot(
        f_band,
        g_profile,
        color=PALETTE["orange"],
        ls="--",
        lw=1.6,
        label="Unsaturated gain profile $g(f) / g_{\\max}$",
    )
    ax.plot(
        cold["freqs"] * 1e-12,
        cold["spec"] ** 2,
        color=PALETTE["grey"],
        ls="-.",
        lw=1.6,
        label=f"Cold-cavity mode ($Q_{{\\mathrm{{cold}}}}={cold['Q_cold']:.1f}$)",
    )
    ax.plot(
        freqs_THz,
        spec_lase,
        color=PALETTE["blue"],
        lw=2.0,
        label="Steady-state laser emission $|E_{\\mathrm{out}}(f)|^2$",
    )
    ax.axvline(
        cold["f_cav"] * 1e-12,
        color=PALETTE["green"],
        ls=":",
        lw=1.4,
        label=f"$f_{{\\mathrm{{cav}}}} = {cold['f_cav']*1e-12:.2f}$ THz",
    )
    ax.set_xlim(210.0, 290.0)
    ax.set_ylim(0.0, 1.34)
    ax.set_xlabel("Frequency $f$ [THz]")
    ax.set_ylabel("Normalized spectral density [-]")
    ax.set_title("(c) Lasing Spectrum vs Cold-Cavity Mode & Gain")
    ax.legend(loc="upper right", framealpha=0.92)

    fig2_path = os.path.join(img_dir, "fabry_perot_laser_dynamics_and_spectrum.png")
    fig2.savefig(fig2_path, dpi=180)
    plt.close(fig2)
    return fig1_path, fig2_path


def main():
    t_start = time.time()
    p = setup_parameters()
    print("=== 1D DBR Fabry-Perot Laser (FourLevelGain) ===")
    print(f"Design wavelength lambda_0 = {p['lam0']*1e6:.3f} um (f_0 = {p['f0']*1e-12:.2f} THz)")
    print(f"DBR mirror reflectance R   = {p['R_mirror']:.5f} ((15/17)^2)")
    print(f"Threshold gain g_th        = {p['g_th']:.4e} 1/m")
    print(f"Threshold inversion dN_th  = {p['dn_th']:.4e} m^-3")
    print(f"Threshold pump rate W_p,th = {p['W_p_th']:.4e} 1/s")
    print(f"Saturation intensity I_sat = {p['I_sat']*1e-6:.3f} MW/m^2")

    cold = run_cold_cavity(p)
    print(
        f"Cold cavity: f_cav = {cold['f_cav']*1e-12:.2f} THz, "
        f"Q_cold = {cold['Q_cold']:.2f}, kappa_cold = {cold['kappa_cold']:.3e} 1/s"
    )

    ratios = np.array([0.0, 0.5, 0.9, 1.5, 2.0, 2.5, 3.0])
    I_shb, dn_shb = [], []
    I_hom, dn_hom = [], []
    traces = {}

    for r in ratios:
        res_shb = run_laser_point(p, r, spatial_averaging=False)
        I_shb.append(res_shb["I_out"])
        dn_shb.append(res_shb["dn"][-1])
        if np.isclose(r, 0.5):
            traces["sub"] = res_shb
        if np.isclose(r, 2.5):
            traces["above_shb"] = res_shb

        res_hom = run_laser_point(p, r, spatial_averaging=True)
        I_hom.append(res_hom["I_out"])
        dn_hom.append(res_hom["dn"][-1])
        if np.isclose(r, 2.5):
            traces["above_hom"] = res_hom

        print(
            f"  W_p/W_p,th = {r:4.2f} | "
            f"I_out (SHB) = {res_shb['I_out']*1e-6:7.3f} MW/m^2 | "
            f"I_out (Hom) = {res_hom['I_out']*1e-6:7.3f} MW/m^2 | "
            f"dN_hom/dN_th = {res_hom['dn'][-1]/p['dn_th']:.4f}"
        )

    sweep_shb = {
        "ratios": ratios,
        "I_out": np.array(I_shb),
        "dn_final": np.array(dn_shb),
    }
    sweep_hom = {
        "ratios": ratios,
        "I_out": np.array(I_hom),
        "dn_final": np.array(dn_hom),
    }

    img_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "img"
    )
    f1, f2 = make_figures(p, cold, sweep_shb, sweep_hom, traces, img_dir)
    print(f"Saved figures:\n  {f1}\n  {f2}")
    print(f"Completed in {time.time() - t_start:.1f} s.")


if __name__ == "__main__":
    main()
