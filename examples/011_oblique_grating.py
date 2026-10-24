# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""Broadband oblique incidence at constant angle on a periodic dielectric grating.

Demonstrates ``wakis.floquet`` (`FloquetCell`, `AngleSweep`, `rcwa_1d_grating`)
to compute diffraction-order efficiencies at a constant oblique angle of
incidence across a broadband wavelength range by resampling a uniform grid of
fixed-``k_par`` Bloch FDTD/FIT simulations.

Produces two figures in ``docs/img/``:
1. ``oblique_floquet_sampling.png``: The ``(k_par, f)`` sampling grid with
   constant-``theta`` curves and the ``m = +1`` and ``m = -2`` Rayleigh
   anomaly cutoffs overlaid (with the legend placed outside the axes).
2. ``oblique_grating_efficiencies.png``: Diffraction order efficiencies and
   the total propagating-order power sum ``sum_m (R_m + T_m)`` versus
   wavelength at constant angle ``theta = 15 deg`` for both Wakis and the 1D
   Fourier Modal Method (RCWA) reference, annotating the energy residual and
   the anomaly-adjacent interpolation points.
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c, epsilon_0

from wakis.floquet import AngleSweep, FloquetCell, rcwa_1d_grating

# ---------------------------------------------------------------------------
# 1. Unit-cell parameters: 1D dielectric lamellar grating in air
# ---------------------------------------------------------------------------
lam0 = 0.30  # center wavelength [m]
f0 = c / lam0  # center frequency ~ 1.0 GHz
period = 1.5 * lam0  # grating period Lx = 0.45 m
h_grating = 0.25 * lam0  # grating tooth height = 0.075 m
eps_ridge = 2.25  # ridge permittivity (n = 1.5, duty cycle 0.5)
ALL_ORDERS = (-2, -1, 0, 1, 2)

freqs = np.linspace(0.86 * f0, 1.06 * f0, 9)
wavelengths = c / freqs
angles_deg = np.array([10.0, 15.0, 20.0])
angles_rad = np.deg2rad(angles_deg)

nx, nz = 24, 120
zmin, zmax = -0.55, 0.55
dz = (zmax - zmin) / nz


def _lamellar_eps(x, z):
    tooth = (np.mod(x, period) < 0.5 * period) & (z > 0.0) & (z < h_grating)
    return np.where(tooth, eps_ridge, 1.0)


def fill_lamellar_grating(solver):
    """Populate the periodic dielectric grating teeth in solver.ieps with subpixel averaging."""
    grid, d = solver.grid, 1e-6 * lam0
    nodes = (grid.x[:-1], grid.y[:-1], grid.z[:-1])
    for ci, comp in enumerate("xyz"):
        pos = [
            (n + np.diff(g) / 2) if k == ci else n
            for k, (n, g) in enumerate(zip(nodes, (grid.x, grid.y, grid.z)))
        ]
        X, _, Z = np.meshgrid(*pos, indexing="ij")
        offsets = [
            (ox, oz)
            for ox in ((0.0,) if ci == 0 else (-d, d))
            for oz in ((0.0,) if ci == 2 else (-d, d))
        ]
        eps = sum(_lamellar_eps(X + ox, Z + oz) for ox, oz in offsets) / len(offsets)
        solver.ieps[:, :, :, comp] = 1.0 / (epsilon_0 * eps)


cell = FloquetCell(
    Lx=period,
    Ly=dz,
    zmin=zmin,
    zmax=zmax,
    nx=nx,
    ny=1,
    nz=nz,
    z_source=-0.30,
    z_refl=-0.16,
    z_tran=0.26,
    z_ref_below=0.0,
    z_ref_above=h_grating,
    eps_below=1.0,
    eps_above=1.0,
    orders_x=ALL_ORDERS,
    orders_y=(0,),
    n_pml=15,
    material_callback=fill_lamellar_grating,
    run_time_cycles=120.0,
)

sweep = AngleSweep(cell, freqs, angles_rad, polarization="TE", n_k=7)


def main():
    # Support batch mode (--list-tasks / --task / --collect) when CLI flags are passed
    if len(sys.argv) > 1:
        sweep.cli()
        return

    result = sweep.run()
    resampled = result.resample()

    # Target angle index 1 -> theta = 15 deg
    ia = 1
    theta_target_deg = angles_deg[ia]
    theta_target_rad = angles_rad[ia]

    # Compute dense and point-matched 1D RCWA references at theta = 15 deg
    freqs_dense = np.linspace(freqs[0], freqs[-1], 65)
    wl_dense = c / freqs_dense
    x_fine = np.linspace(0.0, period, 512, endpoint=False)
    eps_prof = np.where(x_fine < 0.5 * period, eps_ridge, 1.0)
    rcwa_dense = rcwa_1d_grating(
        freqs_dense,
        theta_target_rad,
        period,
        h_grating,
        eps_prof,
        eps_below=1.0,
        eps_above=1.0,
        pol="TE",
        n_harmonics=15,
        orders=ALL_ORDERS,
    )
    rcwa_pts = rcwa_1d_grating(
        freqs,
        theta_target_rad,
        period,
        h_grating,
        eps_prof,
        eps_below=1.0,
        eps_above=1.0,
        pol="TE",
        n_harmonics=15,
        orders=ALL_ORDERS,
    )

    img_dir = Path(__file__).resolve().parent.parent / "docs" / "img"
    img_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Figure 1: (k_par, f) sampling plane with constant-theta rays & cutoffs
    # -----------------------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(7.6, 5.2), dpi=150)
    k_par = result.k_par
    f_ghz = freqs * 1e-9
    im0 = result.orders_x.index(0)

    # Plot background pcolormesh of m = 0 transmission efficiency on (k_par, f)
    t0_k = result.t_order_k[:, :, im0, 0].T  # shape (n_freq, n_k)
    pcm = ax1.pcolormesh(
        k_par,
        f_ghz,
        t0_k,
        shading="nearest",
        cmap="viridis",
        alpha=0.85,
    )
    fig1.colorbar(pcm, ax=ax1, label=r"Raw $T_0(k_\parallel, f)$ on Bloch grid")

    # Vertical lines: fixed-k_par Bloch runs
    for kv in k_par:
        ax1.axvline(kv, color="white", ls="--", lw=0.9, alpha=0.8)
    kk, ff = np.meshgrid(k_par, f_ghz)
    ax1.plot(
        kk.ravel(),
        ff.ravel(),
        "wo",
        ms=4,
        mec="k",
        mew=0.6,
        label=r"Fixed-$k_\parallel$ DFT samples",
    )

    # Constant-angle rays k_par(f) = (2*pi*f/c) sin(theta)
    f_line = np.linspace(0.80 * f0, 1.12 * f0, 200)
    colors = ["#ffdd57", "#ff595e", "#ff924c"]
    for deg, col in zip(angles_deg, colors):
        k_ray = (2.0 * np.pi * f_line / c) * np.sin(np.deg2rad(deg))
        ax1.plot(
            k_ray,
            f_line * 1e-9,
            color=col,
            lw=2.2,
            label=rf"Constant $\theta = {deg:.0f}^\circ$",
        )

    # Rayleigh anomaly cutoffs crossing the sampled (k_par, f) domain:
    # m = +1 cutoff: k_par + 2*pi/P = 2*pi*f/c
    # m = -2 cutoff: 4*pi/P - k_par = 2*pi*f/c
    k_rayleigh_p1 = 2.0 * np.pi * f_line / c - 2.0 * np.pi / period
    k_rayleigh_m2 = 4.0 * np.pi / period - 2.0 * np.pi * f_line / c
    ax1.plot(
        k_rayleigh_p1,
        f_line * 1e-9,
        "c-.",
        lw=1.8,
        label=r"Rayleigh anomaly ($m=+1$ cutoff)",
    )
    ax1.plot(
        k_rayleigh_m2,
        f_line * 1e-9,
        "m:",
        lw=1.8,
        label=r"Rayleigh anomaly ($m=-2$ cutoff)",
    )

    ax1.set_xlim(0.0, max(k_par[-1] * 1.25, 9.5))
    ax1.set_ylim(f_ghz[0] * 0.96, f_ghz[-1] * 1.04)
    ax1.set_xlabel(r"Tangential wavevector $k_\parallel$ [rad/m]")
    ax1.set_ylabel(r"Frequency $f$ [GHz]")
    ax1.set_title(
        r"Broadband oblique-incidence $(k_\parallel, f)$ sampling and constant-$\theta$ rays"
    )
    ax1.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,
        fontsize=8.5,
        framealpha=0.95,
    )
    fig1.tight_layout()
    fig1.subplots_adjust(bottom=0.24)
    fig1_path = img_dir / "oblique_floquet_sampling.png"
    fig1.savefig(fig1_path)
    plt.close(fig1)

    # -----------------------------------------------------------------------
    # Figure 2: Order efficiencies vs wavelength at constant theta = 15 deg
    # -----------------------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(8.0, 5.5), dpi=150)

    order_styles = [
        (-2, "#ff924c", "v", r"$T_{-2}$"),
        (-1, "#1982c4", "o", r"$T_{-1}$"),
        (0, "#8ac926", "s", r"$T_{0}$"),
        (+1, "#ff595e", "^", r"$T_{+1}$"),
    ]
    for m_ord, col, marker, label in order_styles:
        io = ALL_ORDERS.index(m_ord)
        ax2.plot(
            wl_dense,
            rcwa_dense["T_orders"][:, io],
            color=col,
            lw=1.8,
            label=f"{label} (RCWA)",
        )
        ax2.plot(
            wavelengths,
            resampled["T_order"][ia, :, io, 0],
            linestyle="none",
            marker=marker,
            color=col,
            ms=6.5,
            mec="k",
            mew=0.6,
            label=f"{label} (Wakis)",
        )

    # Also plot specular reflection R_0
    ax2.plot(
        wl_dense,
        rcwa_dense["R_orders"][:, im0],
        color="#6a4c93",
        ls="--",
        lw=1.6,
        label=r"$R_0$ (RCWA)",
    )
    ax2.plot(
        wavelengths,
        resampled["R_order"][ia, :, im0, 0],
        linestyle="none",
        marker="D",
        color="#6a4c93",
        ms=5.5,
        mec="k",
        mew=0.6,
        label=r"$R_0$ (Wakis)",
    )

    # Total propagating energy sum_m (R_m + T_m) for BOTH RCWA and Wakis
    rcwa_total_dense = rcwa_dense["R_total"] + rcwa_dense["T_total"]
    rcwa_total_pts = rcwa_pts["R_total"] + rcwa_pts["T_total"]
    wakis_total = resampled["R_total"][ia] + resampled["T_total"][ia]
    rcwa_max_res = float(np.max(np.abs(rcwa_total_pts - 1.0)))
    wakis_max_res = float(np.nanmax(np.abs(wakis_total - 1.0)))

    ax2.plot(
        wl_dense,
        rcwa_total_dense,
        color="#222222",
        ls="-",
        lw=1.4,
        alpha=0.85,
        label=rf"$\sum (R_m+T_m)$ RCWA ($|\Delta|\leq{rcwa_max_res:.1e}$)",
    )
    ax2.plot(
        wavelengths,
        wakis_total,
        "k:",
        marker="+",
        ms=8,
        mew=1.4,
        lw=1.3,
        label=rf"$\sum (R_m+T_m)$ Wakis ($|\Delta|\leq{wakis_max_res:.1e}$)",
    )

    # Mark Rayleigh anomaly wavelengths at theta = 15 deg
    lam_rayleigh_p1 = period * (1.0 - np.sin(theta_target_rad))
    lam_rayleigh_m2 = 0.5 * period * (1.0 + np.sin(theta_target_rad))
    ax2.axvline(
        lam_rayleigh_p1,
        color="gray",
        ls="-.",
        lw=1.2,
        label=rf"$m=+1$ cutoff ($\lambda={lam_rayleigh_p1:.3f}$ m)",
    )
    ax2.axvline(
        lam_rayleigh_m2,
        color="#b56576",
        ls=":",
        lw=1.2,
        label=rf"$m=-2$ cutoff ($\lambda={lam_rayleigh_m2:.3f}$ m)",
    )

    # Annotate anomaly-adjacent interpolated points where non-analytic / resonant
    # curvature reduces cubic interpolation accuracy
    im_m1 = ALL_ORDERS.index(-1)
    im_p1 = ALL_ORDERS.index(+1)
    err_tm1_297 = abs(resampled["T_order"][ia, 6, im_m1, 0] - rcwa_pts["T_orders"][6, im_m1])
    err_tm1_305 = abs(resampled["T_order"][ia, 5, im_m1, 0] - rcwa_pts["T_orders"][5, im_m1])
    err_tp1_305 = abs(resampled["T_order"][ia, 5, im_p1, 0] - rcwa_pts["T_orders"][5, im_p1])
    err_tp1_330 = abs(resampled["T_order"][ia, 2, im_p1, 0] - rcwa_pts["T_orders"][2, im_p1])

    ax2.axvspan(wavelengths[6] - 0.002, wavelengths[5] + 0.002, color="#ffdd57", alpha=0.22)
    ax2.annotate(
        f"Wood/GMR anomaly step\n"
        f"($\\lambda\\approx0.297, 0.305$ m):\n"
        f"$|\\Delta T_{{-1}}|={err_tm1_297:.3f}, {err_tm1_305:.3f}$\n"
        f"$|\\Delta T_{{+1}}|={err_tp1_305:.3f}$",
        xy=(0.5 * (wavelengths[5] + wavelengths[6]), 0.09),
        xytext=(0.289, 0.25),
        fontsize=7.5,
        bbox=dict(boxstyle="round,pad=0.25", fc="#fffdf0", ec="#ccb030", alpha=0.92),
        arrowprops=dict(arrowstyle="->", color="#665500", lw=0.9),
    )
    ax2.annotate(
        f"Cutoff-straddling point\n"
        f"($\\lambda\\approx0.330$ m):\n"
        f"$|\\Delta T_{{+1}}|={err_tp1_330:.3f}$",
        xy=(wavelengths[2], resampled["T_order"][ia, 2, im_p1, 0]),
        xytext=(0.327, 0.22),
        fontsize=7.5,
        bbox=dict(boxstyle="round,pad=0.25", fc="#f7f7f7", ec="#888888", alpha=0.92),
        arrowprops=dict(arrowstyle="->", color="#555555", lw=0.9),
    )

    ax2.set_xlabel(r"Wavelength $\lambda = c / f$ [m]")
    ax2.set_ylabel("Diffraction efficiency")
    ax2.set_ylim(-0.03, 1.09)
    ax2.set_title(
        rf"Lamellar grating efficiencies at constant $\theta = {theta_target_deg:.0f}^\circ$ (TE)"
    )
    ax2.grid(True, ls=":", alpha=0.5)
    ax2.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=4,
        fontsize=7.6,
        framealpha=0.95,
    )
    fig2.tight_layout()
    fig2.subplots_adjust(bottom=0.26)
    fig2_path = img_dir / "oblique_grating_efficiencies.png"
    fig2.savefig(fig2_path)
    plt.close(fig2)

    print(f"Saved {fig1_path} and {fig2_path}")
    print(
        f"Energy conservation at theta={theta_target_deg} deg: "
        f"Wakis min={np.nanmin(wakis_total):.6f}, max={np.nanmax(wakis_total):.6f} "
        f"(max |sum-1|={wakis_max_res:.2e}); "
        f"RCWA max |sum-1|={rcwa_max_res:.2e}"
    )
    for jf in range(freqs.size):
        r_w = resampled["R_order"][ia, jf, :, 0]
        t_w = resampled["T_order"][ia, jf, :, 0]
        r_c = rcwa_pts["R_orders"][jf]
        t_c = rcwa_pts["T_orders"][jf]
        print(
            f"lam={wavelengths[jf]:.5f}m: "
            f"Wakis R={np.round(r_w, 5).tolist()} T={np.round(t_w, 5).tolist()} sum={wakis_total[jf]:.6f} | "
            f"RCWA R={np.round(r_c, 5).tolist()} T={np.round(t_c, 5).tolist()} sum={rcwa_total_pts[jf]:.6f}"
        )


if __name__ == "__main__":
    main()
