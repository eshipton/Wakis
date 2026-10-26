# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""Triangular and honeycomb 2D photonic-crystal bands, unfolding, and valley Berry phase.

Demonstrates :mod:`wakis.lattices` on a Cartesian FIT grid:

1. **Triangular lattice of air holes** in a high-index dielectric slab
   (:math:`r/a = 0.35, \\varepsilon_{\\mathrm{bg}} = 12.0` and
   :math:`r/a = 0.43, \\varepsilon_{\\mathrm{bg}} = 13.0`):
   centred-rectangular supercell (:math:`a \\times \\sqrt{3}a`), primitive-zone
   band unfolding via the internal translation :math:`\\boldsymbol{\\tau} = (a/2, \\sqrt{3}a/2)`,
   complete overlapping TE and TM photonic bandgaps, and 2D FIT (``harminv``)
   vs 2D plane-wave expansion (:class:`~wakis.lattices.PWE2D`) convergence with
   Richardson extrapolation.
2. **Honeycomb lattice of dielectric rods** (:math:`r_0 = 0.18a, \\varepsilon_{\\mathrm{rod}} = 12.0`):
   massless Dirac cone at :math:`K` for inversion-symmetric rods (:math:`\\delta = 0`),
   linear valley-Hall bandgap opening :math:`\\Delta\\omega_K \\propto |\\delta|` under
   inversion symmetry breaking :math:`r_{A,B} = r_0(1 \\pm \\delta)`, and gauge-invariant
   Fukui-Hatsugai-Suzuki Berry curvature :math:`\\Omega_1(\\mathbf{k})` around :math:`K`
   and :math:`K'` from both PWE and 2D FIT.

Usage::

    python examples/016_triangular_honeycomb_lattices.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from wakis import GridFIT3D, SolverFIT3D, figstyle
from wakis.estimate import estimate, preflight
from wakis.lattices import (
    HexagonalLattice,
    PWE2D,
    dirac_berry_curvature,
    fit_berry_curvature_patch,
    fit_supercell_bands_at_k,
)

ROOT = Path(__file__).resolve().parent.parent
IMG_DIR = ROOT / "docs" / "img"
DATA_DIR = Path(__file__).resolve().parent / "data"


def _preflight_supercell(nx: int, ny: int, n_steps: int) -> dict[str, float]:
    """Run Wakis memory/runtime preflight on a 2D supercell FIT grid."""
    est = estimate((nx, ny, 1), steps=n_steps, dtype="complex128", backend="cpu", cpml=False)
    preflight(est, {"ram": 2.0e9, "gpu": 0.0})
    return {"ram_mb": float(est.host_bytes / 1e6), "wall_seconds_est": float(est.seconds)}


def _richardson_second_order(h1: float, val1: float, h2: float, val2: float) -> float:
    """Second-order Richardson extrapolation from two grid spacings h1 > h2."""
    r2 = (h1 / h2) ** 2
    return float((r2 * val2 - val1) / (r2 - 1.0))


def run_all(quick: bool = False) -> dict:
    """Execute the triangular and honeycomb lattice benchmarks and write JSON + figures."""
    t_start = time.perf_counter()
    figstyle.apply_style()
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    lat = HexagonalLattice(a=1.0)
    hs = lat.high_symmetry_points()
    A1, A2 = lat.supercell_vectors

    # ------------------------------------------------------------------
    # 1. Triangular lattice of air holes: supercell unfolding & TE/TM gaps
    # ------------------------------------------------------------------
    n_seg = 8 if quick else 14
    path = lat.bz_path(("Gamma", "M", "K", "Gamma"), n_per_segment=n_seg)
    n_pw_ref = 6 if quick else 8

    # Case A: r/a = 0.43, eps_bg = 13.0 (complete TE + TM photonic bandgap)
    r_gap = 0.43
    eps_gap = 13.0
    pwe_gap_te = PWE2D(lat.a1, lat.a2, lat.triangular_primitive_sites(r_gap, 1.0), eps_bg=eps_gap, n_pw=n_pw_ref, polarization="TE")
    pwe_gap_tm = PWE2D(lat.a1, lat.a2, lat.triangular_primitive_sites(r_gap, 1.0), eps_bg=eps_gap, n_pw=n_pw_ref, polarization="TM")
    bands_gap_te = pwe_gap_te.solve_path(path.k_points, n_bands=4)["frequencies"]
    bands_gap_tm = pwe_gap_tm.solve_path(path.k_points, n_bands=4)["frequencies"]

    te_gap_bot = float(np.max(bands_gap_te[:, 0]))
    te_gap_top = float(np.min(bands_gap_te[:, 1]))
    te_gap_mid = 0.5 * (te_gap_bot + te_gap_top)
    te_gap_ratio = (te_gap_top - te_gap_bot) / te_gap_mid

    tm_gap_bot = float(np.max(bands_gap_tm[:, 1]))
    tm_gap_top = float(np.min(bands_gap_tm[:, 2]))
    tm_gap_mid = 0.5 * (tm_gap_bot + tm_gap_top)
    tm_gap_ratio = (tm_gap_top - tm_gap_bot) / tm_gap_mid

    complete_gap_bot = max(te_gap_bot, tm_gap_bot)
    complete_gap_top = min(te_gap_top, tm_gap_top)
    complete_gap_width = complete_gap_top - complete_gap_bot

    # Case B: r/a = 0.35, eps_bg = 12.0 (supercell unfolding + FIT vs PWE validation)
    r_val = 0.35
    eps_val = 12.0
    pwe_prim_tm = PWE2D(lat.a1, lat.a2, lat.triangular_primitive_sites(r_val, 1.0), eps_bg=eps_val, n_pw=n_pw_ref, polarization="TM")
    pwe_prim_te = PWE2D(lat.a1, lat.a2, lat.triangular_primitive_sites(r_val, 1.0), eps_bg=eps_val, n_pw=n_pw_ref, polarization="TE")
    pwe_sup_tm = PWE2D(A1, A2, lat.triangular_supercell_sites(r_val, 1.0), eps_bg=eps_val, n_pw=n_pw_ref, polarization="TM")
    pwe_sup_te = PWE2D(A1, A2, lat.triangular_supercell_sites(r_val, 1.0), eps_bg=eps_val, n_pw=n_pw_ref, polarization="TE")

    prim_tm_path = pwe_prim_tm.solve_path(path.k_points, n_bands=3)["frequencies"]
    prim_te_path = pwe_prim_te.solve_path(path.k_points, n_bands=3)["frequencies"]
    sup_tm_path = pwe_sup_tm.solve_path(path.k_points, n_bands=6, unfold_supercell=True, lattice_a=1.0)
    sup_te_path = pwe_sup_te.solve_path(path.k_points, n_bands=6, unfold_supercell=True, lattice_a=1.0)

    unfolding_max_diff_tm = float(np.max(np.abs(sup_tm_path["unfolded_frequencies"] - prim_tm_path)))
    unfolding_max_diff_te = float(np.max(np.abs(sup_te_path["unfolded_frequencies"] - prim_te_path)))

    # 2D FIT band validation along high-symmetry k points (sequential runs)
    nx_fit = 20 if quick else 28
    ny_fit = 34 if quick else 48
    n_steps_fit = int(round(80.0 * nx_fit))
    preflight_info = _preflight_supercell(nx_fit, ny_fit, n_steps_fit)

    sample_idx = sorted(set([int(i) for i in np.linspace(2, len(path.s) - 3, 5 if quick else 9)] + list(path.segment_indices[1:-1])))
    fit_path_s = []
    fit_path_tm_b1 = []
    fit_path_te_b1 = []
    inc_sup_val = lat.triangular_supercell_sites(r_val, 1.0)

    for idx in sample_idx:
        k_vec = path.k_points[idx]
        if np.linalg.norm(k_vec) < 1e-6:
            continue
        f_tm_target = float(prim_tm_path[idx, 0])
        f_te_target = float(prim_te_path[idx, 0])
        res_tm = fit_supercell_bands_at_k(
            lat,
            inc_sup_val,
            k_vec,
            polarization="TM",
            nx=nx_fit,
            ny=ny_fit,
            eps_bg=eps_val,
            f_min=max(0.04, f_tm_target - 0.06),
            f_max=f_tm_target + 0.06,
            n_steps=n_steps_fit,
        )
        res_te = fit_supercell_bands_at_k(
            lat,
            inc_sup_val,
            k_vec,
            polarization="TE",
            nx=nx_fit,
            ny=ny_fit,
            eps_bg=eps_val,
            f_min=max(0.04, f_te_target - 0.06),
            f_max=f_te_target + 0.06,
            n_steps=n_steps_fit,
        )
        if res_tm["unfolded_frequencies"].size > 0 and res_te["unfolded_frequencies"].size > 0:
            f_tm_pick = float(res_tm["unfolded_frequencies"][np.argmin(np.abs(res_tm["unfolded_frequencies"] - f_tm_target))])
            f_te_pick = float(res_te["unfolded_frequencies"][np.argmin(np.abs(res_te["unfolded_frequencies"] - f_te_target))])
            fit_path_s.append(float(path.s[idx]))
            fit_path_tm_b1.append(f_tm_pick)
            fit_path_te_b1.append(f_te_pick)

    # Convergence study at K for both PWE (vs n_pw) and 2D FIT (vs nx)
    k_K = hs["K"]
    pwe_ref_fine_tm = float(
        PWE2D(lat.a1, lat.a2, lat.triangular_primitive_sites(r_val, 1.0), eps_bg=eps_val, n_pw=11, polarization="TM").solve_k(k_K, n_bands=2)[0]
    )
    pwe_ref_fine_te = float(
        PWE2D(lat.a1, lat.a2, lat.triangular_primitive_sites(r_val, 1.0), eps_bg=eps_val, n_pw=11, polarization="TE").solve_k(k_K, n_bands=2)[0]
    )

    pwe_npw_list = [3, 4, 5, 6, 8]
    pwe_conv_tm = []
    pwe_conv_te = []
    for npw in pwe_npw_list:
        f_tm_n = float(
            PWE2D(lat.a1, lat.a2, lat.triangular_primitive_sites(r_val, 1.0), eps_bg=eps_val, n_pw=npw, polarization="TM").solve_k(k_K, n_bands=2)[0]
        )
        f_te_n = float(
            PWE2D(lat.a1, lat.a2, lat.triangular_primitive_sites(r_val, 1.0), eps_bg=eps_val, n_pw=npw, polarization="TE").solve_k(k_K, n_bands=2)[0]
        )
        pwe_conv_tm.append(abs(f_tm_n - pwe_ref_fine_tm) / pwe_ref_fine_tm)
        pwe_conv_te.append(abs(f_te_n - pwe_ref_fine_te) / pwe_ref_fine_te)

    fit_grids = [(12, 20), (20, 34), (28, 48)] if quick else [(14, 24), (22, 38), (28, 48), (36, 62)]
    fit_nx_list = [g[0] for g in fit_grids]
    fit_tm_vals = []
    fit_te_vals = []
    for gx, gy in fit_grids:
        nt_g = int(round(82.0 * gx))
        _preflight_supercell(gx, gy, nt_g)
        rtm = fit_supercell_bands_at_k(
            lat, inc_sup_val, k_K, polarization="TM", nx=gx, ny=gy, eps_bg=eps_val, f_min=0.16, f_max=0.27, n_steps=nt_g
        )
        rte = fit_supercell_bands_at_k(
            lat, inc_sup_val, k_K, polarization="TE", nx=gx, ny=gy, eps_bg=eps_val, f_min=0.16, f_max=0.27, n_steps=nt_g
        )
        ftm = float(rtm["unfolded_frequencies"][np.argmin(np.abs(rtm["unfolded_frequencies"] - pwe_ref_fine_tm))])
        fte = float(rte["unfolded_frequencies"][np.argmin(np.abs(rte["unfolded_frequencies"] - pwe_ref_fine_te))])
        fit_tm_vals.append(ftm)
        fit_te_vals.append(fte)

    h_last2 = (1.0 / fit_nx_list[-2], 1.0 / fit_nx_list[-1])
    fit_tm_rich = _richardson_second_order(h_last2[0], fit_tm_vals[-2], h_last2[1], fit_tm_vals[-1])
    fit_te_rich = _richardson_second_order(h_last2[0], fit_te_vals[-2], h_last2[1], fit_te_vals[-1])
    fit_conv_err_tm = [abs(v - fit_tm_rich) / fit_tm_rich for v in fit_tm_vals]
    fit_conv_err_te = [abs(v - fit_te_rich) / fit_te_rich for v in fit_te_vals]

    # Cells per wavelength at finest FIT grid in eps = 12 at f = 0.22 c/a
    min_wavelength_dielectric = 1.0 / (0.35 * np.sqrt(eps_val))
    cells_per_lambda_finest = float(fit_nx_list[-1] * min_wavelength_dielectric)

    # ------------------------------------------------------------------
    # 2. Honeycomb lattice: Dirac point at K, valley-Hall gap, Berry curvature
    # ------------------------------------------------------------------
    r0_hc = 0.18
    eps_hc = 12.0
    pwe_hc_sym = PWE2D(lat.a1, lat.a2, lat.honeycomb_primitive_sites(r0_hc, r0_hc, eps_hc), eps_bg=1.0, n_pw=n_pw_ref, polarization="TM")
    pwe_hc_val = PWE2D(
        lat.a1,
        lat.a2,
        lat.honeycomb_primitive_sites(r0_hc * 1.12, r0_hc * 0.88, eps_hc),
        eps_bg=1.0,
        n_pw=n_pw_ref,
        polarization="TM",
    )
    hc_bands_sym = pwe_hc_sym.solve_path(path.k_points, n_bands=4)["frequencies"]
    hc_bands_val = pwe_hc_val.solve_path(path.k_points, n_bands=4)["frequencies"]

    f_dirac_K = pwe_hc_sym.solve_k(k_K, n_bands=2)
    dirac_freq = float(0.5 * (f_dirac_K[0] + f_dirac_K[1]))
    dirac_splitting_sym = float(abs(f_dirac_K[1] - f_dirac_K[0]))

    # Estimate Dirac velocity v_D / c from slope near K
    dq_v = 0.015 * (2.0 * np.pi)
    f_near_K = pwe_hc_sym.solve_k(k_K + np.array([dq_v, 0.0]), n_bands=2)
    v_d_norm = float(0.5 * (f_near_K[1] - f_near_K[0]) * (2.0 * np.pi) / dq_v)

    deltas_sweep = np.linspace(-0.18, 0.18, 9 if quick else 13)
    signed_gaps = []
    for d in deltas_sweep:
        pwe_d = PWE2D(
            lat.a1,
            lat.a2,
            lat.honeycomb_primitive_sites(r0_hc * (1.0 + d), r0_hc * (1.0 - d), eps_hc),
            eps_bg=1.0,
            n_pw=5 if quick else 7,
            polarization="TM",
        )
        fk = pwe_d.solve_k(k_K, n_bands=2)
        gap_mag = float(fk[1] - fk[0])
        signed_gaps.append(float(np.sign(d) * gap_mag if abs(d) > 1e-12 else 0.0))
    signed_gaps_arr = np.asarray(signed_gaps, dtype=float)
    gap_slope = float(np.dot(deltas_sweep, signed_gaps_arr) / np.dot(deltas_sweep, deltas_sweep))

    # Berry curvature around K and K' for delta = +0.12 and -0.12
    dk_berry = 0.12 * (2.0 * np.pi)
    nk_berry = 9 if quick else 15
    q_berry = np.linspace(-dk_berry, dk_berry, nk_berry)
    pwe_berry_pos = PWE2D(
        lat.a1, lat.a2, lat.honeycomb_primitive_sites(r0_hc * 1.12, r0_hc * 0.88, eps_hc), eps_bg=1.0, n_pw=5, polarization="TM"
    )
    pwe_berry_neg = PWE2D(
        lat.a1, lat.a2, lat.honeycomb_primitive_sites(r0_hc * 0.88, r0_hc * 1.12, eps_hc), eps_bg=1.0, n_pw=5, polarization="TM"
    )
    om_k_pos, c_k_pos = pwe_berry_pos.berry_curvature(k_K[0] + q_berry, k_K[1] + q_berry, band_index=0)
    om_k_neg, c_k_neg = pwe_berry_neg.berry_curvature(k_K[0] + q_berry, k_K[1] + q_berry, band_index=0)
    om_kp_pos, c_kp_pos = pwe_berry_pos.berry_curvature(hs["K'"][0] + q_berry, hs["K'"][1] + q_berry, band_index=0)

    # 2D FIT Berry curvature patch at K for delta = +0.12 and -0.12
    fit_berry_pos = fit_berry_curvature_patch(
        lat, r_0=r0_hc, delta=+0.12, eps_rod=eps_hc, dk_frac=0.08, n_k=3, nx=18, ny=30, n_steps=6000
    )
    fit_berry_neg = fit_berry_curvature_patch(
        lat, r_0=r0_hc, delta=-0.12, eps_rod=eps_hc, dk_frac=0.08, n_k=3, nx=18, ny=30, n_steps=6000
    )

    # ------------------------------------------------------------------
    # Figure 1: Triangular lattice supercell, unfolding, TE/TM gaps, FIT vs PWE convergence
    # ------------------------------------------------------------------
    fig1, axes1 = plt.subplots(2, 2, figsize=(11.5, 8.6))

    # Panel (a): Centred-rectangular supercell schematic vs primitive cell
    ax = axes1[0, 0]
    lx_s, ly_s = lat.supercell_size
    rect = mpatches.Rectangle((0.0, 0.0), lx_s, ly_s, fill=False, edgecolor="#1b6ca8", linewidth=2.0, label=r"Supercell $a \times \sqrt{3}a$")
    ax.add_patch(rect)
    prim_poly = np.array([[0.0, 0.0], lat.a1, lat.a1 + lat.a2, lat.a2, [0.0, 0.0]])
    ax.plot(prim_poly[:, 0], prim_poly[:, 1], "--", color="#d95f02", lw=1.8, label="Primitive rhombic cell")
    for n1 in range(-1, 3):
        for n2 in range(-1, 3):
            pt = n1 * lat.a1 + n2 * lat.a2
            if -0.25 <= pt[0] <= 1.55 and -0.25 <= pt[1] <= 1.95:
                circ = mpatches.Circle((pt[0], pt[1]), radius=r_val, facecolor="#e6f2ff", edgecolor="#1b6ca8", lw=1.1, alpha=0.85)
                ax.add_patch(circ)
                ax.plot(pt[0], pt[1], "o", color="#1b6ca8", ms=4)
    ax.annotate(
        "",
        xy=(lat.tau[0], lat.tau[1]),
        xytext=(0.0, 0.0),
        arrowprops=dict(arrowstyle="->", color="#b2182b", lw=2.0),
    )
    ax.text(0.22, 0.52, r"$\boldsymbol{\tau} = (a/2, \sqrt{3}a/2)$", color="#b2182b", fontsize=9, fontweight="bold")
    ax.set_xlim(-0.25, 1.55)
    ax.set_ylim(-0.25, 1.95)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x / a$")
    ax.set_ylabel(r"$y / a$")
    ax.set_title(r"(a) Centred-rectangular supercell ($L_x=a, L_y=\sqrt{3}a$)")
    ax.legend(loc="upper right", fontsize=8.5)

    # Panel (b): Supercell folded vs unfolded bands + 2D FIT harminv points (r/a = 0.35, eps = 12)
    ax = axes1[0, 1]
    for ib in range(6):
        ax.plot(path.s, sup_tm_path["frequencies"][:, ib], ":", color="#999999", lw=1.0, label="Supercell folded (TM)" if ib == 0 else None)
    for ib in range(3):
        ax.plot(path.s, prim_tm_path[:, ib], "-", color="#1b6ca8", lw=1.8, label="Unfolded primitive TM" if ib == 0 else None)
        ax.plot(path.s, prim_te_path[:, ib], "--", color="#d95f02", lw=1.8, label="Unfolded primitive TE" if ib == 0 else None)
    ax.plot(fit_path_s, fit_path_tm_b1, "o", color="#084594", ms=5.5, label="2D FIT + harminv (TM)")
    ax.plot(fit_path_s, fit_path_te_b1, "s", color="#b30000", ms=5.2, label="2D FIT + harminv (TE)")
    for xpos in path.tick_positions:
        ax.axvline(xpos, color="#cccccc", lw=0.8, zorder=0)
    ax.set_xticks(path.tick_positions)
    ax.set_xticklabels([r"$\Gamma$", r"$M$", r"$K$", r"$\Gamma$"])
    ax.set_xlim(path.s[0], path.s[-1])
    ax.set_ylim(0.0, 0.52)
    ax.set_ylabel(r"Normalized frequency $\omega a / (2\pi c)$")
    ax.set_title(r"(b) Supercell unfolding & 2D FIT validation ($r=0.35a, \varepsilon=12$)")
    ax.legend(loc="upper right", fontsize=8.5)

    # Panel (c): Complete TE + TM photonic bandgap for r/a = 0.43, eps = 13
    ax = axes1[1, 0]
    for ib in range(4):
        ax.plot(path.s, bands_gap_te[:, ib], "-", color="#d95f02", lw=1.8, label="TE ($H_z$)" if ib == 0 else None)
        ax.plot(path.s, bands_gap_tm[:, ib], "--", color="#1b6ca8", lw=1.8, label="TM ($E_z$)" if ib == 0 else None)
    ax.axhspan(te_gap_bot, te_gap_top, color="#fdb863", alpha=0.22, label="TE gap")
    ax.axhspan(complete_gap_bot, complete_gap_top, color="#2ca25f", alpha=0.35, label="Complete TE+TM gap")
    for xpos in path.tick_positions:
        ax.axvline(xpos, color="#cccccc", lw=0.8, zorder=0)
    ax.set_xticks(path.tick_positions)
    ax.set_xticklabels([r"$\Gamma$", r"$M$", r"$K$", r"$\Gamma$"])
    ax.set_xlim(path.s[0], path.s[-1])
    ax.set_ylim(0.0, 0.56)
    ax.set_ylabel(r"Normalized frequency $\omega a / (2\pi c)$")
    ax.set_title(r"(c) Complete TE & TM bandgap ($r=0.43a, \varepsilon=13$)")
    ax.legend(loc="lower right", fontsize=8.5)

    # Panel (d): Log-log convergence of 2D FIT and PWE at K
    ax = axes1[1, 1]
    nx_arr = np.asarray(fit_nx_list, dtype=float)
    ax.loglog(nx_arr, fit_conv_err_tm, "o-", color="#1b6ca8", lw=1.8, label="2D FIT TM error vs $N_x$")
    ax.loglog(nx_arr, fit_conv_err_te, "s-", color="#d95f02", lw=1.8, label="2D FIT TE error vs $N_x$")
    guide_2nd = fit_conv_err_tm[0] * (nx_arr / nx_arr[0]) ** (-2.0)
    ax.loglog(nx_arr, guide_2nd, "k--", lw=1.3, label=r"$\mathcal{O}(N_x^{-2})$ second-order guide")
    npw_arr = np.asarray(pwe_npw_list, dtype=float)
    ax.loglog(npw_arr * 4.0, pwe_conv_tm, "^:", color="#2ca25f", lw=1.4, label=r"PWE TM error vs $4 N_{\mathrm{pw}}$")
    ax.set_xlabel(r"Resolution parameter $N_x$ (cells / $a$)")
    ax.set_ylabel(r"Relative eigenfrequency error at $K$")
    ax.set_title(r"(d) 2D FIT & PWE convergence at $K$ ($r=0.35a, \varepsilon=12$)")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(loc="upper right", fontsize=8.5)

    fig1.tight_layout()
    fig1_path = IMG_DIR / "triangular_lattice_bands_unfolding.png"
    figstyle.save(fig1, fig1_path)
    plt.close(fig1)

    # ------------------------------------------------------------------
    # Figure 2: Honeycomb Dirac cone, valley-Hall gap scaling, Berry curvature
    # ------------------------------------------------------------------
    fig2, axes2 = plt.subplots(1, 3, figsize=(13.2, 4.3))

    # Panel (a): Honeycomb Dirac point (delta=0) vs valley-Hall gap (delta=0.12)
    ax = axes2[0]
    for ib in range(3):
        ax.plot(path.s, hc_bands_sym[:, ib], "--", color="#666666", lw=1.6, label=r"Symmetric ($\delta = 0$, Dirac)" if ib == 0 else None)
        ax.plot(path.s, hc_bands_val[:, ib], "-", color="#b2182b", lw=2.0, label=r"Valley-Hall ($\delta = 0.12$)" if ib == 0 else None)
    f_val_K = pwe_hc_val.solve_k(k_K, n_bands=2)
    ax.axhspan(float(f_val_K[0]), float(f_val_K[1]), color="#f4a582", alpha=0.32, label="Valley-Hall gap")
    for xpos in path.tick_positions:
        ax.axvline(xpos, color="#cccccc", lw=0.8, zorder=0)
    ax.set_xticks(path.tick_positions)
    ax.set_xticklabels([r"$\Gamma$", r"$M$", r"$K$", r"$\Gamma$"])
    ax.set_xlim(path.s[0], path.s[-1])
    ax.set_ylim(0.0, 0.46)
    ax.set_ylabel(r"Normalized frequency $\omega a / (2\pi c)$")
    ax.set_title(r"(a) Honeycomb Dirac point & valley-Hall gap")
    ax.legend(loc="lower right", fontsize=8.5)

    # Panel (b): Signed Dirac gap at K vs inversion asymmetry delta
    ax = axes2[1]
    ax.plot(deltas_sweep, signed_gaps_arr, "o", color="#1b6ca8", ms=5.5, label="PWE eigenmode splitting")
    ax.plot(deltas_sweep, gap_slope * deltas_sweep, "-", color="#d95f02", lw=1.8, label=rf"Linear fit (slope $= {gap_slope:.3f}$)")
    ax.axhline(0.0, color="#888888", lw=0.8, ls=":")
    ax.axvline(0.0, color="#888888", lw=0.8, ls=":")
    ax.set_xlabel(r"Inversion asymmetry $\delta = (r_A - r_B)/(2 r_0)$")
    ax.set_ylabel(r"Signed valley gap $\mathrm{sgn}(\delta)\,\Delta\omega_K a / (2\pi c)$")
    ax.set_title(r"(b) Valley bandgap scaling vs asymmetry $\delta$")
    ax.legend(loc="upper left", fontsize=8.5)

    # Panel (c): Berry curvature cross-section through K and K' (PWE, FIT, Dirac theory)
    ax = axes2[2]
    q_centers = 0.5 * (q_berry[:-1] + q_berry[1:]) / (2.0 * np.pi)
    mid_idx = om_k_pos.shape[1] // 2
    om_dirac_pos = dirac_berry_curvature(
        0.5 * (q_berry[:-1] + q_berry[1:]),
        np.zeros_like(q_centers),
        v_d=v_d_norm,
        delta_omega=float(f_val_K[1] - f_val_K[0]) * 2.0 * np.pi,
        valley=+1,
    )
    om_dirac_neg = -om_dirac_pos
    ax.plot(q_centers, om_k_pos[:, mid_idx], "o-", color="#1b6ca8", lw=1.8, label=rf"PWE at $K$ ($\delta=+0.12, C_K={c_k_pos:+.2f}$)")
    ax.plot(q_centers, om_k_neg[:, mid_idx], "s-", color="#d95f02", lw=1.8, label=rf"PWE at $K$ ($\delta=-0.12, C_K={c_k_neg:+.2f}$)")
    ax.plot(q_centers, om_dirac_pos, "--", color="#084594", lw=1.3, label=r"Massive Dirac ($\delta=+0.12$)")
    ax.plot(q_centers, om_dirac_neg, "--", color="#b30000", lw=1.3, label=r"Massive Dirac ($\delta=-0.12$)")
    # Overlay 2D FIT plaquette values at K
    fit_kx = np.asarray(fit_berry_pos["kx_grid"])
    fit_qc = (0.5 * (fit_kx[:-1] + fit_kx[1:]) - k_K[0]) / (2.0 * np.pi)
    fit_om_pos = np.mean(np.asarray(fit_berry_pos["omega"]), axis=1)
    fit_om_neg = np.mean(np.asarray(fit_berry_neg["omega"]), axis=1)
    ax.plot(fit_qc, fit_om_pos, "D", color="#1b6ca8", ms=6.5, label=rf"2D FIT ($\delta=+0.12, C={float(fit_berry_pos['c_patch']):+.2f}$)")
    ax.plot(fit_qc, fit_om_neg, "D", color="#d95f02", ms=6.5, label=rf"2D FIT ($\delta=-0.12, C={float(fit_berry_neg['c_patch']):+.2f}$)")
    ax.axhline(0.0, color="#888888", lw=0.8, ls=":")
    ax.set_xlabel(r"Wavevector offset $(k_x - K_x) a / (2\pi)$")
    ax.set_ylabel(r"Berry curvature $\Omega_1(\mathbf{k}) / a^2$")
    ax.set_title(r"(c) Valley Berry curvature & sign reversal")
    ax.legend(loc="lower right", fontsize=8.5)

    fig2.tight_layout()
    fig2_path = IMG_DIR / "honeycomb_dirac_and_valley_hall.png"
    figstyle.save(fig2, fig2_path)
    plt.close(fig2)

    summary = {
        "triangular_air_holes_gap": {
            "r_over_a": r_gap,
            "eps_bg": eps_gap,
            "te_gap_bottom": te_gap_bot,
            "te_gap_top": te_gap_top,
            "te_gap_midgap_ratio": float(te_gap_ratio),
            "tm_gap_bottom": tm_gap_bot,
            "tm_gap_top": tm_gap_top,
            "tm_gap_midgap_ratio": float(tm_gap_ratio),
            "complete_gap_bottom": float(complete_gap_bot),
            "complete_gap_top": float(complete_gap_top),
            "complete_gap_width": float(complete_gap_width),
        },
        "triangular_validation_and_unfolding": {
            "r_over_a": r_val,
            "eps_bg": eps_val,
            "unfolding_max_diff_tm": unfolding_max_diff_tm,
            "unfolding_max_diff_te": unfolding_max_diff_te,
            "pwe_ref_K_tm": pwe_ref_fine_tm,
            "pwe_ref_K_te": pwe_ref_fine_te,
            "fit_grids_nx": fit_nx_list,
            "fit_K_tm_values": [float(x) for x in fit_tm_vals],
            "fit_K_te_values": [float(x) for x in fit_te_vals],
            "fit_K_tm_richardson": fit_tm_rich,
            "fit_K_te_richardson": fit_te_rich,
            "fit_vs_pwe_rel_err_tm": float(abs(fit_tm_vals[-1] - pwe_ref_fine_tm) / pwe_ref_fine_tm),
            "fit_vs_pwe_rel_err_te": float(abs(fit_te_vals[-1] - pwe_ref_fine_te) / pwe_ref_fine_te),
            "cells_per_wavelength_finest": cells_per_lambda_finest,
            "preflight": preflight_info,
        },
        "honeycomb_valley_hall": {
            "r0_over_a": r0_hc,
            "eps_rod": eps_hc,
            "dirac_frequency_norm": dirac_freq,
            "dirac_splitting_symmetric": dirac_splitting_sym,
            "dirac_velocity_over_c": v_d_norm,
            "gap_linear_slope": gap_slope,
            "delta_benchmark": 0.12,
            "valley_gap_delta_0p12": float(f_val_K[1] - f_val_K[0]),
            "pwe_patch_chern_K_pos": float(c_k_pos),
            "pwe_patch_chern_K_neg": float(c_k_neg),
            "pwe_patch_chern_Kp_pos": float(c_kp_pos),
            "fit_patch_chern_K_pos": float(fit_berry_pos["c_patch"]),
            "fit_patch_chern_K_neg": float(fit_berry_neg["c_patch"]),
        },
        "wall_time_seconds": float(time.perf_counter() - t_start),
        "figures": [
            str(fig1_path.relative_to(ROOT)),
            str(fig2_path.relative_to(ROOT)),
        ],
    }

    json_path = DATA_DIR / "016_triangular_honeycomb_lattices.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Saved figures: {fig1_path.name}, {fig2_path.name}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Run a faster reduced-resolution sweep.")
    args = parser.parse_args()
    run_all(quick=args.flag if hasattr(args, "flag") else args.quick)
