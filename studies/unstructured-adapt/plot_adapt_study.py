# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""
Plot figures for the adaptive tetrahedral FIT study (`studies/unstructured-adapt/plot_adapt_study.py`).

Usage
-----
    python studies/unstructured-adapt/plot_adapt_study.py --results FILE --outdir DIR

Generates:
    DIR/unstructured_adapt_convergence.png
    DIR/unstructured_adapt_mesh.png
    DIR/unstructured_adapt_effectivity.png
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import numpy as np
from matplotlib.ticker import NullFormatter
from matplotlib.tri import Triangulation

from wakis import figstyle

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


def plot_convergence(data: dict, outdir: str) -> str:
    arms = data.get("arms", {})
    fig, (ax_dof, ax_time) = figstyle.new_figure(
        width_in=7.0, height_in=4.5, ncols=2, nrows=1
    )

    # Panel (a): |k_h^2 - k^2| / k^2 vs free-edge DOFs
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
        label, s_idx = ARM_META.get(name, (name, 0))
        st = figstyle.series_style(s_idx)
        if name in ("random1", "random2"):
            st["alpha"] = 0.5
        ax_dof.loglog(dofs, err, label=label, **st)

        if name == "residual":
            eta_rel = np.array([float(h["eta2_per_m2"] / h["k2_per_m2"]) for h in hist])
            ax_dof.loglog(
                dofs,
                eta_rel,
                color=st["color"],
                linestyle="--",
                linewidth=1.1,
                label=r"$\eta_{\mathrm{res}}^2 / k_h^2$",
            )

    # Reference power-law guides N^(-4/9) and N^(-2/3)
    if "uniform" in arms and arms["uniform"].get("history"):
        u_hist = arms["uniform"]["history"]
        n0 = float(u_hist[0]["n_dofs"])
        n1 = float(u_hist[-1]["n_dofs"])
        e0 = abs(float(u_hist[0]["rel_err_signed"]))
        n_grid = np.geomspace(n0, n1, 50)
        ax_dof.loglog(
            n_grid,
            0.75 * e0 * (n_grid / n0) ** (-4.0 / 9.0),
            color="#555555",
            linestyle=":",
            linewidth=1.1,
            label=r"$N^{-4/9}$ (P1)",
        )
        ax_dof.loglog(
            n_grid,
            0.55 * e0 * (n_grid / n0) ** (-2.0 / 3.0),
            color="#1a1a1a",
            linestyle="-.",
            linewidth=1.1,
            label=r"$N^{-2/3}$ (P2)",
        )

    ax_dof.xaxis.set_minor_formatter(NullFormatter())
    ax_dof.set_xlabel("Free-edge DOFs $N$")
    ax_dof.set_ylabel(r"$|k_h^2 - k^2| / k^2$")
    ax_dof.set_title("Eigenvalue error vs DOFs", pad=10)
    figstyle.panel_label(ax_dof, "(a)")

    # Panel (b): honest wall-time comparison (cumulative adaptive vs single uniform solves)
    for name in ("uniform", "residual", "zz", "geometric", "random0"):
        if name not in arms:
            continue
        hist = arms[name].get("history", [])
        if not hist:
            continue
        err = np.array([max(abs(float(h["rel_err_signed"])), 1e-16) for h in hist])
        _, s_idx = ARM_META.get(name, (name, 0))
        st = figstyle.series_style(s_idx)
        if name == "uniform":
            t_single = np.array([float(h["t_total_s"]) for h in hist])
            ax_time.loglog(
                t_single,
                err,
                label="Uniform (single solve)",
                **st,
            )
        else:
            t_cum = np.array([float(h["cum_wall_s"]) for h in hist])
            short_lab = {
                "residual": "Residual (cum. loop)",
                "zz": "ZZ curl (cum. loop)",
                "geometric": "Geometric (cum. loop)",
                "random0": "Random (cum. loop)",
            }.get(name, name)
            ax_time.loglog(t_cum, err, label=short_lab, **st)

    p6 = data.get("p6_voronoi_export", {})
    if p6 and "residual_final" in p6 and "uniform_closest_error" in p6:
        dt_ratio = p6.get("dt_ratio_adapt_over_uniform", np.nan)
        wr_ratio = p6.get("work_rate_ratio_adapt_over_uniform", np.nan)
        txt = f"P6 step ratio: {dt_ratio:.2f}\n(N/dt) ratio: {wr_ratio:.2f}"
        ax_time.text(
            0.04,
            0.05,
            txt,
            transform=ax_time.transAxes,
            fontsize=8.5,
            va="bottom",
            ha="left",
            bbox={
                "boxstyle": "round,pad=0.25",
                "fc": "#f8f9fa",
                "ec": "#cccccc",
                "alpha": 0.9,
            },
        )

    ax_time.xaxis.set_minor_formatter(NullFormatter())
    ax_time.set_xlabel("Wall time [s]")
    ax_time.set_ylabel(r"$|k_h^2 - k^2| / k^2$")
    ax_time.set_title("Error vs wall time", pad=10)
    figstyle.panel_label(ax_time, "(b)")

    figstyle.place_legend_below(ax_dof, ncol=2, y_offset=-0.24)
    figstyle.place_legend_below(ax_time, ncol=1, y_offset=-0.24)

    out_path = os.path.join(outdir, "unstructured_adapt_convergence.png")
    figstyle.save(fig, out_path, close=True, auto_legend_below=False)
    return out_path


def plot_mesh(data: dict, outdir: str) -> str:
    res_arm = data.get("arms", {}).get("residual", {})
    top = res_arm.get("top_face")
    if not top:
        raise ValueError(
            "Merged JSON is missing arms.residual.top_face for mesh plotting."
        )

    xy = np.asarray(top["xy_m"], dtype=float)
    tris = np.asarray(top["triangles"], dtype=int)
    h_t = np.asarray(top["h_T_m"], dtype=float)
    eta_t = np.asarray(top["eta_T_per_m"], dtype=float)
    tri_obj = Triangulation(xy[:, 0], xy[:, 1], tris)

    fig, (ax_h, ax_eta, ax_zoom) = figstyle.new_figure(
        width_in=7.0, height_in=2.8, ncols=3, nrows=1
    )

    # Panel (a): log10(h_T [m]) on top face z = d
    log_h = np.log10(np.maximum(h_t, 1e-12))
    im_h = ax_h.tripcolor(tri_obj, facecolors=log_h, cmap="viridis", edgecolors="none")
    ax_h.triplot(tri_obj, color="#1a1a1a", linewidth=0.2, alpha=0.45, marker="none")
    ax_h.plot(
        0.0,
        0.0,
        marker="o",
        markersize=5,
        color="#D55E00",
        markeredgecolor="#1a1a1a",
        linestyle="none",
    )
    # 0.5 m scale bar in the missing quadrant [0.2, 0.7] at y = -0.78
    ax_h.plot(
        [0.2, 0.7],
        [-0.78, -0.78],
        color="#1a1a1a",
        linewidth=2.0,
        marker="none",
    )
    ax_h.text(0.45, -0.68, "0.5 m", ha="center", va="bottom", fontsize=8.5)
    ax_h.set_aspect("equal")
    ax_h.set_xlim(-1.05, 1.05)
    ax_h.set_ylim(-1.05, 1.05)
    ax_h.set_xlabel("$x$ [m]")
    ax_h.set_ylabel("$y$ [m]")
    ax_h.set_title(r"Size $\log_{10}(h_T/\mathrm{m})$", pad=10)
    cb_h = fig.colorbar(im_h, ax=ax_h, fraction=0.046, pad=0.04)
    cb_h.ax.tick_params(labelsize=8.5)
    figstyle.panel_label(ax_h, "(a)")

    # Panel (b): log10(eta_T [1/m]) on the same top face
    log_eta = np.log10(np.maximum(eta_t, 1e-12))
    im_eta = ax_eta.tripcolor(
        tri_obj, facecolors=log_eta, cmap="magma", edgecolors="none"
    )
    ax_eta.triplot(tri_obj, color="#ffffff", linewidth=0.15, alpha=0.3, marker="none")
    ax_eta.plot(
        0.0,
        0.0,
        marker="o",
        markersize=5,
        color="#56B4E9",
        markeredgecolor="#1a1a1a",
        linestyle="none",
    )
    ax_eta.set_aspect("equal")
    ax_eta.set_xlim(-1.05, 1.05)
    ax_eta.set_ylim(-1.05, 1.05)
    ax_eta.set_xlabel("$x$ [m]")
    ax_eta.set_title(r"Indicator $\log_{10}(\eta_T\,\mathrm{m})$", pad=10)
    cb_eta = fig.colorbar(im_eta, ax=ax_eta, fraction=0.046, pad=0.04)
    cb_eta.ax.tick_params(labelsize=8.5)
    figstyle.panel_label(ax_eta, "(b)")

    # Panel (c): Zoom near the re-entrant corner (0, 0)
    ax_zoom.tripcolor(
        tri_obj,
        facecolors=log_h,
        cmap="viridis",
        vmin=float(np.min(log_h)),
        vmax=float(np.max(log_h)),
        edgecolors="none",
    )
    ax_zoom.triplot(tri_obj, color="#1a1a1a", linewidth=0.45, marker="none")
    ax_zoom.plot(
        0.0,
        0.0,
        marker="o",
        markersize=6,
        color="#D55E00",
        markeredgecolor="#1a1a1a",
        linestyle="none",
    )
    ax_zoom.set_aspect("equal")
    ax_zoom.set_xlim(-0.25, 0.25)
    ax_zoom.set_ylim(-0.25, 0.25)
    ax_zoom.set_xlabel("$x$ [m]")
    ax_zoom.set_title("Corner zoom", pad=10)
    figstyle.panel_label(ax_zoom, "(c)")

    out_path = os.path.join(outdir, "unstructured_adapt_mesh.png")
    figstyle.save(fig, out_path, close=True, auto_legend_below=False)
    return out_path


def plot_effectivity(data: dict, outdir: str) -> str:
    arms = data.get("arms", {})
    fig, (ax_box, ax_lprism) = figstyle.new_figure(
        width_in=7.0, height_in=3.3, ncols=2, nrows=1
    )

    # Panel (a): Box effectivity indices theta_res and theta_ZZ vs DOFs
    box_levels = arms.get("box", {}).get("levels", [])
    if box_levels:
        dofs = np.array([float(lv["n_dofs"]) for lv in box_levels])
        th_res = np.array([float(lv["theta_res"]) for lv in box_levels])
        th_zz = np.array([float(lv["theta_zz"]) for lv in box_levels])
        ax_box.semilogx(
            dofs,
            th_res,
            label=r"$\theta_{\mathrm{res}} = \eta_{\mathrm{res}} / |||u - u_h|||$",
            **figstyle.series_style(0),
        )
        ax_box.semilogx(
            dofs,
            th_zz,
            label=r"$\theta_{\mathrm{ZZ}} = \eta_{\mathrm{ZZ}} / \|\mu^{-1/2}\nabla\times(u - u_h)\|$",
            **figstyle.series_style(1),
        )
        ax_box.axhline(
            1.0,
            color="#555555",
            linestyle=":",
            linewidth=0.9,
            label="Ideal (1.0)",
        )
    ax_box.xaxis.set_minor_formatter(NullFormatter())
    ax_box.set_xlabel("Free-edge DOFs $N$")
    ax_box.set_ylabel(r"Effectivity $\theta$")
    ax_box.set_title("Box TM110 effectivity (E3)", pad=10)
    figstyle.panel_label(ax_box, "(a)")

    # Panel (b): L-prism eigenvalue effectivity eta^2 / |k_h^2 - k^2|
    for idx, name in enumerate(("uniform", "residual", "zz")):
        if name not in arms:
            continue
        hist = arms[name].get("history", [])
        if not hist:
            continue
        dofs = np.array([float(h["n_dofs"]) for h in hist])
        eig_eff = np.array(
            [
                float(
                    h["eta2_per_m2"]
                    / max(abs(h["k2_per_m2"] * h["rel_err_signed"]), 1e-30)
                )
                for h in hist
            ]
        )
        label, s_idx = ARM_META.get(name, (name, idx))
        ax_lprism.loglog(dofs, eig_eff, label=label, **figstyle.series_style(s_idx))

    ax_lprism.xaxis.set_minor_formatter(NullFormatter())
    ax_lprism.set_xlabel("Free-edge DOFs $N$")
    ax_lprism.set_ylabel(r"$\eta_{\mathrm{res}}^2 / |k_h^2 - k^2|$")
    ax_lprism.set_title(r"L-prism ratio $\eta^2 / |k_h^2 - k^2|$", pad=10)
    figstyle.panel_label(ax_lprism, "(b)")

    figstyle.place_legend_below(ax_box, ncol=1, y_offset=-0.24)
    figstyle.place_legend_below(ax_lprism, ncol=1, y_offset=-0.24)

    out_path = os.path.join(outdir, "unstructured_adapt_effectivity.png")
    figstyle.save(fig, out_path, close=True, auto_legend_below=False)
    return out_path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Plot adaptive tetrahedral FIT study figures."
    )
    parser.add_argument(
        "--results",
        required=True,
        help="Path to merged study JSON file.",
    )
    parser.add_argument(
        "--outdir",
        required=True,
        help="Output directory for PNG figures.",
    )
    args = parser.parse_args(argv)

    with open(args.results, encoding="utf-8") as fh:
        data = json.load(fh)

    os.makedirs(args.outdir, exist_ok=True)
    with figstyle.apply_style():
        p1 = plot_convergence(data, args.outdir)
        p2 = plot_mesh(data, args.outdir)
        p3 = plot_effectivity(data, args.outdir)

    print(f"[plot_adapt_study] wrote:\n  {p1}\n  {p2}\n  {p3}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
