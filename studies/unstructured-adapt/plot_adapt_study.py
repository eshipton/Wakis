# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""
Plot figures for the adaptive tetrahedral FIT study (`studies/unstructured-adapt/plot_adapt_study.py`).

Usage
-----
    python studies/unstructured-adapt/plot_adapt_study.py [--results FILE] [--outdir DIR]

Generates:
    DIR/unstructured_adapt_mesh.png
    DIR/unstructured_adapt_convergence.png
    DIR/unstructured_adapt_effectivity.png
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm
from matplotlib.patches import Rectangle
from matplotlib.ticker import NullFormatter
from matplotlib.tri import Triangulation

from wakis import figstyle

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = Path(__file__).resolve().parent / "results" / "adapt_study.json"
DEFAULT_OUTDIR = REPO_ROOT / "docs" / "img"

ARM_META = {
    "uniform": ("Uniform", 0),
    "residual": (r"Residual ($\theta=0.5$)", 1),
    "zz": (r"ZZ curl ($\theta=0.5$)", 2),
    "geometric": ("Geometric", 3),
    "random0": ("Random (3 seeds)", 4),
    "random1": ("_nolegend_", 4),
    "random2": ("_nolegend_", 4),
    "theta03": (r"Residual ($\theta=0.3$)", 5),
    "theta07": (r"Residual ($\theta=0.7$)", 6),
}


def plot_mesh(data: dict, outdir: str) -> str:
    """
    Plot (a) initial L-prism mesh cut view (z = 0.2 m, level 0),
    (b) final residual-adapted L-prism mesh cut view coloured by bisection
    refinement level l_T, and (c) zoom near the re-entrant corner (0, 0).
    """
    res_arm = data.get("arms", {}).get("residual", {})
    init_cut = res_arm.get("initial_mesh_cut")
    adapt_cut = res_arm.get("adapted_mesh_cut") or res_arm.get("top_face")
    if not adapt_cut:
        raise ValueError(
            "Merged JSON is missing arms.residual.adapted_mesh_cut / top_face."
        )

    xy_ad = np.asarray(adapt_cut["xy_m"], dtype=float)
    tris_ad = np.asarray(adapt_cut["triangles"], dtype=int)
    tri_ad = Triangulation(xy_ad[:, 0], xy_ad[:, 1], tris_ad)
    if "level_T" in adapt_cut:
        lvl_ad = np.asarray(adapt_cut["level_T"], dtype=float)
    else:
        h_ad = np.asarray(adapt_cut["h_T_m"], dtype=float)
        h0 = float(np.max(h_ad))
        lvl_ad = np.rint(3.0 * np.log2(h0 / np.maximum(h_ad, 1e-12)))

    if init_cut:
        xy_in = np.asarray(init_cut["xy_m"], dtype=float)
        tris_in = np.asarray(init_cut["triangles"], dtype=int)
        tri_in = Triangulation(xy_in[:, 0], xy_in[:, 1], tris_in)
        lvl_in = np.asarray(
            init_cut.get("level_T", np.zeros(len(tris_in))), dtype=float
        )
        n_cells_in = int(init_cut.get("n_cells_3d", 1350))
        n_dofs_in = int(init_cut.get("n_dofs_3d", 1193))
        z_cut = float(init_cut.get("z_m", 0.2))
    else:
        tri_in = tri_ad
        lvl_in = np.zeros_like(lvl_ad)
        n_cells_in = 1350
        n_dofs_in = 1193
        z_cut = float(adapt_cut.get("z_m", 0.2))

    n_cells_ad = int(adapt_cut.get("n_cells_3d", res_arm.get("n_cells_final", 37020)))
    n_dofs_ad = int(adapt_cut.get("n_dofs_3d", res_arm.get("n_dofs_final", 41817)))
    l_max = int(max(np.max(lvl_ad), adapt_cut.get("level_max_3d", int(np.max(lvl_ad)))))

    boundaries = np.arange(-0.5, l_max + 1.5, 1.0)
    cmap = plt.get_cmap("viridis", l_max + 1)
    norm = BoundaryNorm(boundaries, cmap.N)

    fig, (ax_in, ax_ad, ax_zoom) = plt.subplots(
        1,
        3,
        figsize=(7.4, 2.85),
        gridspec_kw={
            "wspace": 0.28,
            "left": 0.07,
            "right": 0.90,
            "bottom": 0.17,
            "top": 0.82,
        },
    )

    # Panel (a): Initial mesh cut view at z = z_cut
    ax_in.tripcolor(
        tri_in,
        facecolors=lvl_in,
        cmap=cmap,
        norm=norm,
        edgecolors="none",
    )
    ax_in.triplot(tri_in, color="#e0e0e0", linewidth=0.38, alpha=0.85, marker="none")
    ax_in.plot(
        0.0,
        0.0,
        marker="o",
        markersize=5,
        color="#D55E00",
        markeredgecolor="#ffffff",
        markeredgewidth=0.7,
        linestyle="none",
        zorder=5,
    )
    # 0.5 m scale bar in missing quadrant [0, 1] x [-1, 0]
    ax_in.plot(
        [0.2, 0.7], [-0.78, -0.78], color="#1a1a1a", linewidth=2.0, marker="none"
    )
    ax_in.text(0.45, -0.68, "0.5 m", ha="center", va="bottom", fontsize=8.0)
    ax_in.text(
        0.05,
        -0.30,
        f"{n_cells_in:,} tets\n{n_dofs_in:,} DOFs\n$\\ell_T = 0$",
        fontsize=7.8,
        ha="left",
        va="center",
        bbox={
            "boxstyle": "round,pad=0.22",
            "fc": "#f8f9fa",
            "ec": "#cccccc",
            "alpha": 0.92,
        },
    )
    ax_in.set_aspect("equal")
    ax_in.set_xlim(-1.05, 1.05)
    ax_in.set_ylim(-1.05, 1.05)
    ax_in.set_xlabel("$x$ [m]")
    ax_in.set_ylabel("$y$ [m]")
    ax_in.set_title(f"Initial mesh ($z = {z_cut:.1f}$ m)", pad=13, fontsize=9.2)
    figstyle.panel_label(ax_in, "(a)")

    # Panel (b): Adapted mesh cut view coloured by bisection refinement level l_T
    im_ad = ax_ad.tripcolor(
        tri_ad,
        facecolors=lvl_ad,
        cmap=cmap,
        norm=norm,
        edgecolors="none",
    )
    ax_ad.triplot(tri_ad, color="#1a1a1a", linewidth=0.18, alpha=0.45, marker="none")
    ax_ad.plot(
        0.0,
        0.0,
        marker="o",
        markersize=5,
        color="#D55E00",
        markeredgecolor="#ffffff",
        markeredgewidth=0.7,
        linestyle="none",
        zorder=5,
    )
    rect = Rectangle(
        (-0.25, -0.25),
        0.50,
        0.50,
        fill=False,
        edgecolor="#D55E00",
        linestyle="--",
        linewidth=1.1,
        zorder=4,
    )
    ax_ad.add_patch(rect)
    ax_ad.text(
        0.05,
        -0.38,
        f"{n_cells_ad:,} tets\n{n_dofs_ad:,} DOFs\n$\\ell_T \\in [0, {l_max}]$",
        fontsize=7.8,
        ha="left",
        va="center",
        bbox={
            "boxstyle": "round,pad=0.22",
            "fc": "#f8f9fa",
            "ec": "#cccccc",
            "alpha": 0.92,
        },
    )
    ax_ad.set_aspect("equal")
    ax_ad.set_xlim(-1.05, 1.05)
    ax_ad.set_ylim(-1.05, 1.05)
    ax_ad.set_xlabel("$x$ [m]")
    ax_ad.set_title(f"Adapted mesh ($z = {z_cut:.1f}$ m)", pad=13, fontsize=9.2)
    figstyle.panel_label(ax_ad, "(b)")

    # Panel (c): Corner zoom [-0.25, 0.25]^2 coloured by bisection level l_T
    ax_zoom.tripcolor(
        tri_ad,
        facecolors=lvl_ad,
        cmap=cmap,
        norm=norm,
        edgecolors="none",
    )
    ax_zoom.triplot(tri_ad, color="#1a1a1a", linewidth=0.38, alpha=0.65, marker="none")
    ax_zoom.plot(
        0.0,
        0.0,
        marker="o",
        markersize=5.5,
        color="#D55E00",
        markeredgecolor="#ffffff",
        markeredgewidth=0.8,
        linestyle="none",
        zorder=5,
    )
    ax_zoom.set_aspect("equal")
    ax_zoom.set_xlim(-0.25, 0.25)
    ax_zoom.set_ylim(-0.25, 0.25)
    ax_zoom.set_xticks([-0.2, 0.0, 0.2])
    ax_zoom.set_yticks([-0.2, 0.0, 0.2])
    ax_zoom.set_xlabel("$x$ [m]")
    ax_zoom.set_title("Re-entrant corner zoom", pad=13, fontsize=9.2)
    figstyle.panel_label(ax_zoom, "(c)")

    # Dedicated colorbar axis on the right
    cbar_ax = fig.add_axes([0.915, 0.19, 0.018, 0.64])
    ticks = list(range(0, l_max + 1, 2 if l_max >= 8 else 1))
    cb = fig.colorbar(im_ad, cax=cbar_ax, ticks=ticks)
    cb.set_label(r"Bisection level $\ell_T = \log_2(V_0/V_T)$", fontsize=8.5)
    cb.ax.tick_params(labelsize=8.0)

    out_path = os.path.join(outdir, "unstructured_adapt_mesh.png")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def plot_convergence(data: dict, outdir: str) -> str:
    """
    Plot (a) energy error |||e|||^2 and indicator eta^2 vs DOFs for adaptive
    (residual, ZZ) vs uniform with N^(-4/9) and N^(-2/3) guides, and
    (b) relative eigenvalue error |k_h^2 - k^2|/k^2 vs DOFs across all arms.
    """
    arms = data.get("arms", {})
    fig, (ax_en, ax_eig) = figstyle.new_figure(
        width_in=7.4, height_in=4.8, ncols=2, nrows=1
    )
    fig.subplots_adjust(wspace=0.34, top=0.86)

    # Panel (a): Energy error |||e|||^2 and indicator eta^2 vs free-edge DOFs N
    for name in ("uniform", "residual", "zz", "geometric", "random0"):
        if name not in arms:
            continue
        hist = arms[name].get("history", [])
        if not hist or "energy_err2_per_m2" not in hist[0]:
            continue
        dofs = np.array([float(h["n_dofs"]) for h in hist])
        en2 = np.array([float(h["energy_err2_per_m2"]) for h in hist])
        label, s_idx = ARM_META[name]
        st = figstyle.series_style(s_idx)
        short_label = "Random (seed 0)" if name == "random0" else label
        ax_en.loglog(
            dofs,
            en2,
            label=rf"$|||\mathbf{{e}}|||^2$ {short_label}",
            **st,
        )

    # Plot indicator eta^2 for Uniform, Residual, and ZZ (open markers, dashed)
    for name, eta_key, eta_lab in (
        ("uniform", "eta2_per_m2", r"$\eta_{\mathrm{res}}^2$ Uniform"),
        ("residual", "eta2_per_m2", r"$\eta_{\mathrm{res}}^2$ Residual"),
        ("zz", "eta2_zz_per_m2", r"$\eta_{\mathrm{ZZ}}^2$ ZZ curl"),
    ):
        if name not in arms:
            continue
        hist = arms[name].get("history", [])
        if not hist:
            continue
        dofs = np.array([float(h["n_dofs"]) for h in hist])
        eta2 = np.array([float(h[eta_key]) for h in hist])
        _, s_idx = ARM_META[name]
        st = figstyle.series_style(s_idx)
        ax_en.loglog(
            dofs,
            eta2,
            color=st["color"],
            linestyle="--",
            marker=st.get("marker", "o"),
            markerfacecolor="none",
            markersize=4.5,
            linewidth=1.1,
            label=eta_lab,
        )

    # Reference guides N^(-4/9) and N^(-2/3) anchored on energy error and indicator
    if "uniform" in arms and arms["uniform"].get("history"):
        u_hist = arms["uniform"]["history"]
        n0 = float(u_hist[0]["n_dofs"])
        n1 = float(u_hist[-1]["n_dofs"])
        n_grid = np.geomspace(n0, n1, 50)
        if "energy_err2_per_m2" in u_hist[0]:
            en0 = float(u_hist[0]["energy_err2_per_m2"])
            ax_en.loglog(
                n_grid,
                0.88 * en0 * (n_grid / n0) ** (-4.0 / 9.0),
                color="#555555",
                linestyle=":",
                linewidth=1.15,
                label=r"$N^{-4/9}$ (P1 uniform)",
            )
            ax_en.loglog(
                n_grid,
                0.68 * en0 * (n_grid / n0) ** (-2.0 / 3.0),
                color="#1a1a1a",
                linestyle="-.",
                linewidth=1.15,
                label=r"$N^{-2/3}$ (P2 adaptive)",
            )
        eta0 = float(u_hist[0]["eta2_per_m2"])
        ax_en.loglog(
            n_grid,
            0.90 * eta0 * (n_grid / n0) ** (-4.0 / 9.0),
            color="#555555",
            linestyle=":",
            linewidth=1.0,
            label="_nolegend_",
        )
        ax_en.loglog(
            n_grid,
            0.72 * eta0 * (n_grid / n0) ** (-2.0 / 3.0),
            color="#1a1a1a",
            linestyle="-.",
            linewidth=1.0,
            label="_nolegend_",
        )

    # Annotate measured A2' slopes on Panel (a)
    u_en_s = arms.get("uniform", {}).get("slope_fit_energy", {}).get("slope")
    r_en_s = arms.get("residual", {}).get("slope_fit_energy", {}).get("slope")
    u_eta_s = arms.get("uniform", {}).get("slope_fit_eta2", {}).get("slope")
    r_eta_s = arms.get("residual", {}).get("slope_fit_eta2", {}).get("slope")
    if u_en_s is not None and r_en_s is not None:
        slope_txt = (
            f"Res.: $|||\\mathbf{{e}}|||^2$ {r_en_s:.3f}, $\\eta^2$ {r_eta_s:.3f}\n"
            f"Unif.: $|||\\mathbf{{e}}|||^2$ {u_en_s:.3f}, $\\eta^2$ {u_eta_s:.3f}"
        )
        ax_en.text(
            0.03,
            0.52,
            slope_txt,
            transform=ax_en.transAxes,
            fontsize=7.2,
            va="center",
            ha="left",
            bbox={
                "boxstyle": "round,pad=0.22",
                "fc": "#f8f9fa",
                "ec": "#cccccc",
                "alpha": 0.92,
            },
        )

    ax_en.xaxis.set_minor_formatter(NullFormatter())
    ax_en.set_xlabel("Free-edge DOFs $N$")
    ax_en.set_ylabel(r"$|||\mathbf{e}|||^2,\;\eta^2\;\;[\mathrm{m}^{-2}]$")
    ax_en.set_title(
        r"Energy error $|||\mathbf{e}|||^2$ & $\eta^2$ (A2$^\prime$)", pad=13
    )
    figstyle.panel_label(ax_en, "(a)")

    # Panel (b): Relative eigenvalue error |k_h^2 - k^2| / k^2 vs free-edge DOFs N
    for name in (
        "uniform",
        "residual",
        "zz",
        "geometric",
        "random0",
        "random1",
        "random2",
        "theta03",
        "theta07",
    ):
        if name not in arms:
            continue
        hist = arms[name].get("history", [])
        if not hist:
            continue
        dofs = np.array([float(h["n_dofs"]) for h in hist])
        err = np.array([max(abs(float(h["rel_err_signed"])), 1e-16) for h in hist])
        label, s_idx = ARM_META[name]
        st = figstyle.series_style(s_idx)
        if name in ("random1", "random2"):
            st["alpha"] = 0.5
        ax_eig.loglog(dofs, err, label=label, **st)

    if "uniform" in arms and arms["uniform"].get("history"):
        u_hist = arms["uniform"]["history"]
        n0 = float(u_hist[0]["n_dofs"])
        n1 = float(u_hist[-1]["n_dofs"])
        e0 = abs(float(u_hist[0]["rel_err_signed"]))
        n_grid = np.geomspace(n0, n1, 50)
        ax_eig.loglog(
            n_grid,
            0.75 * e0 * (n_grid / n0) ** (-4.0 / 9.0),
            color="#555555",
            linestyle=":",
            linewidth=1.15,
            label=r"$N^{-4/9}$ (P1)",
        )
        ax_eig.loglog(
            n_grid,
            0.55 * e0 * (n_grid / n0) ** (-2.0 / 3.0),
            color="#1a1a1a",
            linestyle="-.",
            linewidth=1.15,
            label=r"$N^{-2/3}$ (P2)",
        )

    eq = data.get("equal_dof_ratios", {}).get("vs_uniform", {}).get("residual", {})
    if eq:
        r_eig = float(eq.get("ratio_uniform_over_arm", 1.711))
        r_en = float(eq.get("energy_ratio_uniform_over_arm", 1.291))
        n_eq = int(eq.get("target_dofs", 41817))
        ax_eig.text(
            0.03,
            0.05,
            f"At $N={n_eq:,}$: Unif/Res\n$|\\Delta k^2|$: ${r_eig:.2f}\\times$, $|||\\mathbf{{e}}|||^2$: ${r_en:.2f}\\times$",
            transform=ax_eig.transAxes,
            fontsize=7.3,
            va="bottom",
            ha="left",
            bbox={
                "boxstyle": "round,pad=0.22",
                "fc": "#f8f9fa",
                "ec": "#cccccc",
                "alpha": 0.92,
            },
        )

    ax_eig.xaxis.set_minor_formatter(NullFormatter())
    ax_eig.set_xlabel("Free-edge DOFs $N$")
    ax_eig.set_ylabel(r"$|k_h^2 - k^2| / k^2$")
    ax_eig.set_title("Eigenvalue error vs DOFs (A1/A2)", pad=13)
    figstyle.panel_label(ax_eig, "(b)")

    figstyle.place_legend_below(ax_en, ncol=2, y_offset=-0.23, fontsize=7.6)
    figstyle.place_legend_below(ax_eig, ncol=2, y_offset=-0.23, fontsize=7.6)

    out_path = os.path.join(outdir, "unstructured_adapt_convergence.png")
    figstyle.save(fig, out_path, close=True, auto_legend_below=False)
    return out_path


def plot_effectivity(data: dict, outdir: str) -> str:
    """
    Plot (a) energy-norm effectivity indices theta_res = eta_res / |||e||| and
    theta_ZZ = eta_ZZ / ||curl e|| vs DOFs N on the PEC box (E3) and L-prism,
    and (b) L-prism indicator-to-eigenvalue error ratio eta_res^2 / |k_h^2 - k^2|.
    """
    arms = data.get("arms", {})
    fig, (ax_eff, ax_eig_ratio) = figstyle.new_figure(
        width_in=7.4, height_in=4.0, ncols=2, nrows=1
    )
    fig.subplots_adjust(wspace=0.34, top=0.85)

    # Panel (a): Effectivity indices theta_res and theta_ZZ vs DOFs (Box + L-prism)
    box_levels = arms.get("box", {}).get("levels", [])
    if box_levels:
        dofs_b = np.array([float(lv["n_dofs"]) for lv in box_levels])
        th_res_b = np.array([float(lv["theta_res"]) for lv in box_levels])
        th_zz_b = np.array([float(lv["theta_zz"]) for lv in box_levels])
        st_b0 = figstyle.series_style(0)
        st_b1 = figstyle.series_style(1)
        ax_eff.semilogx(
            dofs_b,
            th_res_b,
            color=st_b0["color"],
            linestyle="-",
            marker="s",
            markersize=5.5,
            linewidth=1.5,
            label=r"Box $\theta_{\mathrm{res}}$",
        )
        ax_eff.semilogx(
            dofs_b,
            th_zz_b,
            color=st_b1["color"],
            linestyle="-",
            marker="D",
            markersize=5.0,
            linewidth=1.5,
            label=r"Box $\theta_{\mathrm{ZZ}}$",
        )

    for name, s_idx, short_name in (
        ("uniform", 2, "L-prism Unif."),
        ("residual", 3, "L-prism Res."),
        ("zz", 4, "L-prism ZZ"),
    ):
        hist = arms.get(name, {}).get("history", [])
        if not hist or "theta_res" not in hist[0]:
            continue
        dofs = np.array([float(h["n_dofs"]) for h in hist])
        th_res = np.array([float(h["theta_res"]) for h in hist])
        th_zz = np.array([float(h["theta_zz"]) for h in hist])
        st = figstyle.series_style(s_idx)
        ax_eff.semilogx(
            dofs,
            th_res,
            color=st["color"],
            linestyle="--",
            marker=st.get("marker", "o"),
            markersize=4.5,
            linewidth=1.25,
            label=rf"{short_name} $\theta_{{\mathrm{{res}}}}$",
        )
        ax_eff.semilogx(
            dofs,
            th_zz,
            color=st["color"],
            linestyle=":",
            marker=st.get("marker", "o"),
            markerfacecolor="none",
            markersize=4.5,
            linewidth=1.2,
            label=rf"{short_name} $\theta_{{\mathrm{{ZZ}}}}$",
        )

    ax_eff.axhline(
        1.0,
        color="#555555",
        linestyle="-.",
        linewidth=0.95,
        label="Ideal (1.0)",
    )
    ax_eff.xaxis.set_minor_formatter(NullFormatter())
    ax_eff.set_xlabel("Free-edge DOFs $N$")
    ax_eff.set_ylabel(r"Effectivity index $\theta$")
    ax_eff.set_title(
        r"Effectivity $\theta_{\mathrm{res}}$ and $\theta_{\mathrm{ZZ}}$ vs DOFs",
        pad=13,
    )
    figstyle.panel_label(ax_eff, "(a)")

    # Panel (b): L-prism eigenvalue ratio eta_res^2 / |k_h^2 - k^2|
    for name in ("uniform", "residual", "zz", "geometric"):
        if name not in arms:
            continue
        hist = arms[name].get("history", [])
        if not hist:
            continue
        dofs = np.array([float(h["n_dofs"]) for h in hist])
        eig_eff = np.array([float(h["eigenvalue_effectivity"]) for h in hist])
        label, s_idx = ARM_META[name]
        ax_eig_ratio.loglog(dofs, eig_eff, label=label, **figstyle.series_style(s_idx))

    ax_eig_ratio.axhline(
        1.0,
        color="#555555",
        linestyle=":",
        linewidth=0.95,
        label=r"Reliability bound ($\geq 1$)",
    )
    ax_eig_ratio.xaxis.set_minor_formatter(NullFormatter())
    ax_eig_ratio.set_xlabel("Free-edge DOFs $N$")
    ax_eig_ratio.set_ylabel(r"$\eta_{\mathrm{res}}^2 / |k_h^2 - k^2|$")
    ax_eig_ratio.set_title(
        r"L-prism ratio $\eta_{\mathrm{res}}^2 / |k_h^2 - k^2|$", pad=13
    )
    figstyle.panel_label(ax_eig_ratio, "(b)")

    figstyle.place_legend_below(ax_eff, ncol=2, y_offset=-0.23, fontsize=7.8)
    figstyle.place_legend_below(ax_eig_ratio, ncol=2, y_offset=-0.23, fontsize=7.8)

    out_path = os.path.join(outdir, "unstructured_adapt_effectivity.png")
    figstyle.save(fig, out_path, close=True, auto_legend_below=False)
    return out_path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Plot adaptive tetrahedral FIT study figures."
    )
    parser.add_argument(
        "--results",
        default=str(DEFAULT_RESULTS),
        help="Path to merged study JSON file.",
    )
    parser.add_argument(
        "--outdir",
        default=str(DEFAULT_OUTDIR),
        help="Output directory for PNG figures.",
    )
    args = parser.parse_args(argv)

    with open(args.results, encoding="utf-8") as fh:
        data = json.load(fh)

    os.makedirs(args.outdir, exist_ok=True)
    with figstyle.apply_style():
        p1 = plot_mesh(data, args.outdir)
        p2 = plot_convergence(data, args.outdir)
        p3 = plot_effectivity(data, args.outdir)

    print(f"[plot_adapt_study] wrote:\n  {p1}\n  {p2}\n  {p3}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
