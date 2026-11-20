# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Plot Wave-4 curvilinear-hexahedral FIT diagnostic figures from the JSON outputs
of ``bn_wall_diagnosis.py`` and ``perturbed_box_order.py``:

1. ``curvilinear3d_bn_wall_diagnosis.png``:
   (a) Log-log convergence of the 1-wall octant magnetic normal residual
   ``m1 = |B_c . n_v| / ||B_c||`` and the facet-to-node normal angle bound
   ``g = ||n_v x n_f||`` vs resolution ``N`` across ``sphere_nodes``,
   ``cubed_sphere_ball_nodes``, and ``butterfly_disk_nodes`` with a ``1/N``
   order guide.
   (b) Log-log decay of the smooth-wall cavity eigenmode node-averaged normal
   residual ``m3 = |B_v . n| / max_v ||B_v||`` (lowest TM and TE modes,
   evaluated against both ``n_v`` and ``n_exact``) vs ``N`` with ``O(N^-1)``,
   ``O(N^-2)``, and ``O(N^-3)`` order guides.
2. ``curvilinear3d_perturbed_box_order.png``:
   Log-log ``TE101`` eigenfrequency error decomposition
   ``e(amp) = e_Yee(0) + e_dist`` into the negative second-order Yee dispersion
   term ``|-e_Yee|``, the positive mesh-distortion term ``+e_dist``, and the
   total magnitude ``|e(amp)|`` across refinement families ``F1`` (including the
   pre-fix ``_pec_reconstruction="drop_only"`` comparison), ``F2``, and ``F3``
   with order guides.

Usage
-----
    python studies/curvilinear3d/plot_w4_diagnostics.py
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.ticker import NullFormatter, ScalarFormatter

matplotlib.use("Agg")

from wakis import figstyle

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

_DEFAULT_OUT_DIR = Path(__file__).resolve().parents[2] / "docs" / "img"
_DEFAULT_BN_JSON = (
    Path(__file__).resolve().parent / "results" / "bn_wall_diagnosis.json"
)
_DEFAULT_PB_JSON = (
    Path(__file__).resolve().parent / "results" / "perturbed_box_order.json"
)


def _format_int_log_xaxis(ax, ticks):
    """Format a log-scale x-axis with clean integer tick labels."""
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="x", which="minor", bottom=False)


def plot_bn_wall_diagnosis(bn_data: dict, out_path: Path) -> Path:
    """Render Figure (a): ``m1`` vs ``g`` vs ``N`` and cavity eigenmode ``m3`` decay.

    Parameters
    ----------
    bn_data : dict
        Parsed JSON dictionary from ``bn_wall_diagnosis.py``.
    out_path : Path
        Destination PNG file path.

    Returns
    -------
    Path
        Saved PNG file path.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    families = [
        ("sphere_nodes", "Sphere (1-blk)", "#0072B2", "o"),
        ("cubed_sphere_ball_nodes", "Cubed sphere", "#D55E00", "s"),
        ("butterfly_disk_nodes", "Butterfly cyl.", "#009E73", "^"),
    ]

    with figstyle.apply_style():
        fig, axes = figstyle.new_figure(
            width_in=7.2,
            height_in=4.3,
            ncols=2,
            nrows=1,
            gridspec_kw={"wspace": 0.36},
        )
        ax_m1, ax_m3 = axes

        # ------------------------------------------------------------------
        # Panel (a): 1-wall octant m1 vs facet-node angle bound g vs N
        # ------------------------------------------------------------------
        n_all = []
        for key, label, color, marker in families:
            fam = bn_data[key]
            runs = fam["runs"]
            ns = np.array([float(r["n"]) for r in runs], dtype=float)
            m1_vals = np.array(
                [float(r["octants"]["one_wall_octants"]["max_m1"]) for r in runs],
                dtype=float,
            )
            g_vals = np.array(
                [float(r["octants"]["one_wall_octants"]["max_g"]) for r in runs],
                dtype=float,
            )
            ord_m1 = float(fam["orders"]["one_wall_octants"]["max_m1"]["ls_order"])
            n_all.extend(ns.tolist())

            ax_m1.loglog(
                ns,
                m1_vals,
                color=color,
                linestyle="-",
                marker=marker,
                linewidth=1.8,
                markersize=5.4,
                markerfacecolor=color,
                markeredgecolor=color,
                label=f"{label}: m₁ (ord {ord_m1:.2f})",
                zorder=4,
            )
            ax_m1.loglog(
                ns,
                g_vals,
                color=color,
                linestyle="--",
                marker=marker,
                linewidth=1.4,
                markersize=5.2,
                markerfacecolor="white",
                markeredgecolor=color,
                markeredgewidth=1.3,
                label=f"{label}: g = ‖n_v × n_f‖",
                zorder=3,
            )

        n_ref = np.geomspace(min(n_all), max(n_all), 32)
        g_ref = 0.38 * (n_ref / min(n_all)) ** (-1.0)
        ax_m1.loglog(
            n_ref,
            g_ref,
            color="#1A1A1A",
            linestyle=":",
            marker="none",
            linewidth=1.4,
            label="Order guide: O(N⁻¹)",
            zorder=2,
        )

        max_m0 = float(bn_data.get("max_m0_globally", 0.0))
        ax_m1.text(
            0.04,
            0.05,
            f"Facet normal: max m₀ = {max_m0:.2e}\nIdentity: m₁ ≤ g on all octants",
            transform=ax_m1.transAxes,
            fontsize=8.5,
            color="#1A1A1A",
            va="bottom",
            ha="left",
        )

        figstyle.spectrum_grid(ax_m1)
        _format_int_log_xaxis(ax_m1, [4, 6, 8, 12, 16])
        ax_m1.set_xlabel("Cells per block edge N [-]")
        ax_m1.set_ylabel("Max 1-wall residual m₁ and bound g [-]")
        ax_m1.set_title("(a) Octant m₁ vs angle bound g", fontsize=10.0)
        figstyle.place_legend_below(ax_m1, ncol=1, y_offset=-0.20, fontsize=8.8)

        # ------------------------------------------------------------------
        # Panel (b): Cavity eigenmode smooth-wall node residual m3 decay vs N
        # ------------------------------------------------------------------
        m3_specs = [
            (
                "cubed_sphere_ball_nodes",
                "TM_lowest",
                "max_m3_nv",
                "Cubed sph. TM (n_v)",
                "#D55E00",
                "-",
                "s",
                True,
            ),
            (
                "cubed_sphere_ball_nodes",
                "TE_lowest",
                "max_m3_nv",
                "Cubed sph. TE (n_v)",
                "#D55E00",
                "--",
                "s",
                False,
            ),
            (
                "cubed_sphere_ball_nodes",
                "TE_lowest",
                "max_m3_exact",
                "Cubed sph. TE (n_ex)",
                "#CC79A7",
                "-.",
                "D",
                False,
            ),
            (
                "butterfly_disk_nodes",
                "TM_lowest",
                "max_m3_nv",
                "Butterfly TM010",
                "#009E73",
                "-",
                "^",
                True,
            ),
            (
                "butterfly_disk_nodes",
                "TE_lowest",
                "max_m3_nv",
                "Butterfly TE111",
                "#009E73",
                "--",
                "^",
                False,
            ),
            (
                "sphere_nodes",
                "TM_lowest",
                "max_m3_nv",
                "Sphere (1-blk) TM",
                "#0072B2",
                "-",
                "o",
                True,
            ),
            (
                "sphere_nodes",
                "TE_lowest",
                "max_m3_nv",
                "Sphere (1-blk) TE",
                "#0072B2",
                "--",
                "o",
                False,
            ),
        ]

        for (
            fam_key,
            mode_key,
            metric_key,
            lbl,
            color,
            ls,
            marker,
            filled,
        ) in m3_specs:
            fam = bn_data[fam_key]
            runs = fam["runs"]
            ns = np.array([float(r["n"]) for r in runs], dtype=float)
            vals = np.array(
                [
                    float(r["m3"][mode_key]["per_class"]["smooth_wall"][metric_key])
                    for r in runs
                ],
                dtype=float,
            )
            ord_key = f"{mode_key}_{metric_key.replace('max_', '')}_ls_order"
            ord_val = float(fam["orders"]["per_class"]["smooth_wall"][ord_key])
            ax_m3.loglog(
                ns,
                vals,
                color=color,
                linestyle=ls,
                marker=marker,
                linewidth=1.7,
                markersize=5.0,
                markerfacecolor=color if filled else "white",
                markeredgecolor=color,
                markeredgewidth=1.3,
                label=f"{lbl} (ord {ord_val:.2f})",
            )

        ax_m3.loglog(
            n_ref,
            0.032 * (n_ref / min(n_all)) ** (-1.0),
            color="#4B5563",
            linestyle=":",
            marker="none",
            linewidth=1.3,
            label="Guide: O(N⁻¹)",
        )
        n_ref_cs = np.geomspace(4.0, 8.0, 24)
        ax_m3.loglog(
            n_ref_cs,
            0.095 * (n_ref_cs / 4.0) ** (-2.0),
            color="#1A1A1A",
            linestyle="-.",
            marker="none",
            linewidth=1.3,
            label="Guide: O(N⁻²)",
        )
        ax_m3.loglog(
            n_ref_cs,
            0.0042 * (n_ref_cs / 4.0) ** (-3.0),
            color="#8C6D00",
            linestyle=(0, (3, 1, 1, 1)),
            marker="none",
            linewidth=1.3,
            label="Guide: O(N⁻³)",
        )

        figstyle.spectrum_grid(ax_m3)
        _format_int_log_xaxis(ax_m3, [4, 6, 8, 12, 16])
        ax_m3.set_xlabel("Cells per block edge N [-]")
        ax_m3.set_ylabel("Smooth-wall eigenmode residual m₃ [-]")
        ax_m3.set_title("(b) Eigenmode m₃ decay on smooth walls", fontsize=10.0)
        figstyle.place_legend_below(ax_m3, ncol=1, y_offset=-0.20, fontsize=8.8)

        figstyle.save(fig, out_path, close=True, auto_legend_below=False)

    return out_path


def plot_perturbed_box_order(pb_data: dict, out_path: Path) -> Path:
    """Render Figure (b): perturbed-box ``TE101`` error decomposition vs ``n``.

    Parameters
    ----------
    pb_data : dict
        Parsed JSON dictionary from ``perturbed_box_order.py``.
    out_path : Path
        Destination PNG file path.

    Returns
    -------
    Path
        Saved PNG file path.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    panel_specs = [
        (
            "F1",
            "(a) F1: (n, n, n) cell",
            3.5,
            True,
            [4, 6, 8, 12, 16, 24],
        ),
        (
            "F2",
            "(b) F2: (4k, 3k, 4k) cell",
            3.5,
            False,
            [4, 8, 12, 16, 24],
        ),
        (
            "F3",
            "(c) F3: (n, n, n) fixed map",
            1.5,
            False,
            [6, 8, 12, 16, 24],
        ),
    ]

    with figstyle.apply_style():
        fig, axes = figstyle.new_figure(
            width_in=7.4,
            height_in=4.35,
            ncols=3,
            nrows=1,
            gridspec_kw={"wspace": 0.36},
        )

        for ax, (fkey, title, dist_guide_order, show_drop_only, xticks) in zip(
            axes, panel_specs
        ):
            fam = pb_data[fkey]
            ns = np.array(fam["ns"], dtype=float)
            e_amp = np.abs(np.array(fam["signed_errors_amp"], dtype=float))
            e_yee = np.abs(np.array(fam["signed_errors_yee_amp0"], dtype=float))
            e_dist = np.abs(np.array(fam["signed_errors_dist"], dtype=float))

            ord_amp = float(fam["ls_order_amp"])
            ord_yee = float(fam["ls_order_yee_amp0"])
            ord_dist = float(fam["ls_order_dist"])

            ax.loglog(
                ns,
                e_amp,
                color="#0072B2",
                linestyle="-",
                marker="o",
                linewidth=1.8,
                markersize=5.0,
                markerfacecolor="#0072B2",
                label=f"|e(amp)| (LS {ord_amp:.2f})",
                zorder=5,
            )
            ax.loglog(
                ns,
                e_yee,
                color="#009E73",
                linestyle="--",
                marker="s",
                linewidth=1.6,
                markersize=4.8,
                markerfacecolor="white",
                markeredgecolor="#009E73",
                markeredgewidth=1.3,
                label=f"|-e_Yee(0)| (LS {ord_yee:.2f})",
                zorder=4,
            )
            ax.loglog(
                ns,
                e_dist,
                color="#D55E00",
                linestyle="-",
                marker="^",
                linewidth=1.8,
                markersize=5.0,
                markerfacecolor="#D55E00",
                label=f"+e_dist (LS {ord_dist:.2f})",
                zorder=5,
            )

            if show_drop_only and "F1_study_window_drop_only" in pb_data:
                drop = pb_data["F1_study_window_drop_only"]
                ns_d = np.array(drop["ns"], dtype=float)
                e_amp_d = np.abs(np.array(drop["signed_errors_amp"], dtype=float))
                e_dist_d = np.abs(np.array(drop["signed_errors_dist"], dtype=float))
                d_drop = float(drop["D_drop_only_ls_order_amp"])
                ord_dist_d = float(drop["ls_order_dist"])
                ax.loglog(
                    ns_d,
                    e_amp_d,
                    color="#CC79A7",
                    linestyle="-.",
                    marker="D",
                    linewidth=1.5,
                    markersize=4.6,
                    markerfacecolor="white",
                    markeredgecolor="#CC79A7",
                    markeredgewidth=1.2,
                    label=f"|e| drop_only (D={d_drop:.2f})",
                    zorder=3,
                )
                ax.loglog(
                    ns_d,
                    e_dist_d,
                    color="#8C6D00",
                    linestyle=":",
                    marker="v",
                    linewidth=1.6,
                    markersize=4.8,
                    markerfacecolor="#8C6D00",
                    label=f"+e_dist drop ({ord_dist_d:.2f})",
                    zorder=4,
                )

            n_ref = np.geomspace(ns[0], ns[-1], 32)
            g2 = 1.45 * e_yee[0] * (n_ref / ns[0]) ** (-2.0)
            ax.loglog(
                n_ref,
                g2,
                color="#1A1A1A",
                linestyle=":",
                marker="none",
                linewidth=1.3,
                label="Guide: O(n⁻²)",
                zorder=2,
            )

            n0_idx = 1 if (fkey == "F1" and len(ns) > 1) else 0
            n_ref_d = np.geomspace(ns[n0_idx], ns[-1], 32)
            gd = (
                0.72
                * e_dist[n0_idx]
                * (n_ref_d / ns[n0_idx]) ** (-float(dist_guide_order))
            )
            guide_lbl = (
                "Guide: O(n⁻³·⁵)" if dist_guide_order == 3.5 else "Guide: O(n⁻¹·⁵)"
            )
            ax.loglog(
                n_ref_d,
                gd,
                color="#4B5563",
                linestyle="-.",
                marker="none",
                linewidth=1.3,
                label=guide_lbl,
                zorder=2,
            )

            figstyle.spectrum_grid(ax)
            _format_int_log_xaxis(ax, xticks)
            ax.set_xlabel("Grid resolution n [-]")
            if fkey == "F1":
                ax.set_ylabel("Relative frequency error |Δω / ω_ana| [-]")
            ax.set_title(title, fontsize=9.8)
            figstyle.place_legend_below(ax, ncol=1, y_offset=-0.20, fontsize=8.6)

        figstyle.save(fig, out_path, close=True, auto_legend_below=False)

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bn-json",
        type=Path,
        default=_DEFAULT_BN_JSON,
    )
    parser.add_argument(
        "--pb-json",
        type=Path,
        default=_DEFAULT_PB_JSON,
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_DEFAULT_OUT_DIR,
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.bn_json.exists():
        bn_data = json.loads(args.bn_json.read_text())
        p1 = plot_bn_wall_diagnosis(
            bn_data, args.out_dir / "curvilinear3d_bn_wall_diagnosis.png"
        )
        print(f"Wrote {p1}")
    if args.pb_json.exists():
        pb_data = json.loads(args.pb_json.read_text())
        p2 = plot_perturbed_box_order(
            pb_data, args.out_dir / "curvilinear3d_perturbed_box_order.png"
        )
        print(f"Wrote {p2}")


if __name__ == "__main__":
    main()
