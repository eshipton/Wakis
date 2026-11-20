# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Plot projected Steiner repair figures from ``studies/unstructured-cadmesh/results/steiner.json``:

1. ``steiner_rim_zoom.png``:
   2-panel ``(x, y)`` zoom of the stuck negative Voronoi face ``(0, 1, 43)``
   next to the curved cylinder wall ``r = a = 0.1 m`` before and after
   projected Steiner edge splitting, showing the exact CAD circle ``r = a``,
   the unprojected chord midpoint ``m`` (sagitta ``2.748 mm`` inside ``r = a``),
   and the projected Steiner node ``p`` on ``r = a`` (radial error
   ``1.39e-17 m``).
2. ``steiner_dt_ratios.png``:
   (a) Solver guard negative-face count and maximum stable time step
   ``dt_max`` [ps] per repair path on the cylinder-rim benchmark
   (``repair=False``, ``wave-3 flips``, ``w4 auto``, ``w4 CAD projector`` at
   ``n_sweeps=0`` and ``n_sweeps=6``, and ``analytic``).
   (b) Repaired-to-unrepaired ``dt_max`` ratio and negative-face counts across
   the 6-case CAD benchmark suite (``pillbox``, ``layered``, ``muon`` at
   ``optimize="none"`` and ``optimize="gmsh"``, ``cpw=10``).

Usage
-----
    python studies/unstructured-cadmesh/plot_steiner.py
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.patches import Polygon

matplotlib.use("Agg")

from wakis import figstyle

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

HERE = Path(__file__).resolve().parent
_DEFAULT_JSON = HERE / "results" / "steiner.json"
_DEFAULT_OUT_DIR = HERE.parents[1] / "docs" / "img"


def _draw_tet_edges_xy(ax, tets_xyz, color, lw=1.0, alpha=0.85, ls="-", label=None):
    edges_seen = set()
    first = True
    for tet in tets_xyz:
        pts = np.asarray(tet, dtype=float)
        for i in range(4):
            for j in range(i + 1, 4):
                p0 = tuple(np.round(pts[i, :2], 9))
                p1 = tuple(np.round(pts[j, :2], 9))
                key = (min(p0, p1), max(p0, p1))
                if key in edges_seen:
                    continue
                edges_seen.add(key)
                ax.plot(
                    [pts[i, 0], pts[j, 0]],
                    [pts[i, 1], pts[j, 1]],
                    color=color,
                    lw=lw,
                    alpha=alpha,
                    linestyle=ls,
                    marker="none",
                    label=label if first else None,
                    zorder=3,
                )
                first = False


def plot_rim_zoom(data: dict, out_path: Path) -> Path:
    """Render Figure (a): before/after ``(x, y)`` zoom near the curved CAD rim."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    zoom = data.get("zoom_example")
    if not zoom:
        raise ValueError("No zoom_example in steiner.json")

    R = float(zoom["radius_m"])
    face_pts = np.asarray(zoom["stuck_face_xyz_m"], dtype=float)
    edge_pts = np.asarray(zoom["split_edge_xyz_m"], dtype=float)
    chord_mid = np.asarray(zoom["chord_midpoint_xyz_m"], dtype=float)
    steiner_pts = np.asarray(zoom["steiner_nodes_xyz_m"], dtype=float)
    tets_before = zoom["tets_before_xyz_m"]
    tets_after = zoom["tets_after_xyz_m"]

    all_xy = np.vstack([face_pts[:, :2], edge_pts[:, :2]])
    if len(steiner_pts):
        all_xy = np.vstack([all_xy, steiner_pts[:, :2]])
    for tet in tets_before + tets_after:
        all_xy = np.vstack([all_xy, np.asarray(tet, dtype=float)[:, :2]])

    xmin, ymin = all_xy.min(axis=0) - 0.0045
    xmax, ymax = all_xy.max(axis=0) + 0.0045

    th0 = np.arctan2(max(ymin, -R), max(xmin, 1e-6)) - 0.08
    th1 = np.arctan2(min(ymax, R), max(xmin, 1e-6)) + 0.08
    thetas = np.linspace(th0, th1, 300)
    arc_x = R * np.cos(thetas)
    arc_y = R * np.sin(thetas)

    with figstyle.apply_style():
        fig, axes = figstyle.new_figure(
            width_in=7.2,
            height_in=4.05,
            ncols=2,
            nrows=1,
            gridspec_kw={"wspace": 0.28},
        )
        ax_b, ax_a = axes

        # ------------------------------------------------------------------
        # Panel (a): Before repair (wave-3 flips refused on locked rim chord)
        # ------------------------------------------------------------------
        ax_b.plot(
            arc_x,
            arc_y,
            color="#1A1A1A",
            linestyle="--",
            linewidth=1.5,
            marker="none",
            label="Exact CAD circle r = a (0.1 m)",
            zorder=2,
        )
        _draw_tet_edges_xy(
            ax_b,
            tets_before,
            color="#6B7280",
            lw=1.1,
            alpha=0.85,
            label="Incident tetrahedra (before)",
        )
        poly_b = Polygon(
            face_pts[:, :2],
            closed=True,
            facecolor="#D55E00",
            edgecolor="#D55E00",
            alpha=0.24,
            lw=1.8,
            label="Negative dual face (0, 1, 43)",
            zorder=4,
        )
        ax_b.add_patch(poly_b)
        ax_b.plot(
            edge_pts[:, 0],
            edge_pts[:, 1],
            color="#0072B2",
            lw=2.6,
            marker="o",
            markersize=5.5,
            label="Locked rim chord e = (0, 1)",
            zorder=5,
        )
        ax_b.plot(
            [face_pts[2, 0]],
            [face_pts[2, 1]],
            color="#1A1A1A",
            marker="s",
            linestyle="none",
            markersize=5.2,
            label="Encroaching interior node v₄₃",
            zorder=6,
        )
        ax_b.plot(
            [chord_mid[0]],
            [chord_mid[1]],
            color="#D55E00",
            marker="X",
            linestyle="none",
            markersize=7.5,
            markeredgecolor="#1A1A1A",
            markeredgewidth=0.6,
            label="Chord midpoint m (off CAD by 2.75 mm)",
            zorder=7,
        )

        ax_b.text(
            edge_pts[0, 0] - 0.0038,
            edge_pts[0, 1] + 0.0012,
            "v₀",
            fontsize=9.0,
            fontweight="bold",
            color="#0072B2",
        )
        ax_b.text(
            edge_pts[1, 0] - 0.0035,
            edge_pts[1, 1] + 0.0010,
            "v₁",
            fontsize=9.0,
            fontweight="bold",
            color="#0072B2",
        )
        ax_b.text(
            face_pts[2, 0] - 0.0035,
            face_pts[2, 1] - 0.0035,
            "v₄₃",
            fontsize=9.0,
            fontweight="bold",
            color="#1A1A1A",
        )

        # 5 mm physical scale bar
        sb_x0, sb_y0 = xmin + 0.002, ymin + 0.003
        ax_b.plot(
            [sb_x0, sb_x0 + 0.005],
            [sb_y0, sb_y0],
            color="#1A1A1A",
            lw=2.2,
            marker="none",
            zorder=8,
        )
        ax_b.text(
            sb_x0 + 0.0025,
            sb_y0 + 0.0012,
            "5 mm",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="#1A1A1A",
        )

        ax_b.set_title("(a) Before repair (1 negative face)", fontsize=10.0)
        ax_b.set_xlabel("x [m]")
        ax_b.set_ylabel("y [m]")
        ax_b.set_xlim(xmin, xmax)
        ax_b.set_ylim(ymin, ymax)
        ax_b.set_aspect("equal", adjustable="box")
        figstyle.spectrum_grid(ax_b)
        figstyle.place_legend_below(ax_b, ncol=1, y_offset=-0.20, fontsize=8.6)

        # ------------------------------------------------------------------
        # Panel (b): After projected Steiner split onto CAD circle r = a
        # ------------------------------------------------------------------
        ax_a.plot(
            arc_x,
            arc_y,
            color="#1A1A1A",
            linestyle="--",
            linewidth=1.5,
            marker="none",
            label="Exact CAD circle r = a (0.1 m)",
            zorder=2,
        )
        _draw_tet_edges_xy(
            ax_a,
            tets_after,
            color="#009E73",
            lw=1.25,
            alpha=0.9,
            label="4 sub-tetrahedra (all positive vol.)",
        )
        if len(steiner_pts):
            p_st = steiner_pts[0]
            ax_a.plot(
                [edge_pts[0, 0], p_st[0], edge_pts[1, 0]],
                [edge_pts[0, 1], p_st[1], edge_pts[1, 1]],
                color="#0072B2",
                lw=2.3,
                marker="o",
                markersize=5.0,
                label="Split rim sub-chords (v₀–p, p–v₁)",
                zorder=5,
            )
            ax_a.plot(
                [chord_mid[0]],
                [chord_mid[1]],
                color="#D55E00",
                marker="X",
                linestyle="none",
                markersize=7.0,
                alpha=0.8,
                label="Rejected chord midpoint m",
                zorder=6,
            )
            ax_a.annotate(
                "",
                xy=(p_st[0], p_st[1]),
                xytext=(chord_mid[0], chord_mid[1]),
                arrowprops={
                    "arrowstyle": "->",
                    "color": "#D55E00",
                    "lw": 1.5,
                },
                zorder=7,
            )
            ax_a.plot(
                [p_st[0]],
                [p_st[1]],
                color="#009E73",
                marker="*",
                linestyle="none",
                markersize=11.5,
                markeredgecolor="#1A1A1A",
                markeredgewidth=0.7,
                label="Projected Steiner node p on r = a",
                zorder=8,
            )
            ax_a.text(
                p_st[0] - 0.0052,
                p_st[1] + 0.0014,
                "p (v₄₄)",
                fontsize=9.0,
                fontweight="bold",
                color="#009E73",
            )

        ax_a.text(
            xmin + 0.0015,
            ymin + 0.0020,
            "|r(p) − a| = 1.39e-17 m\n0 negative faces after split",
            fontsize=8.5,
            color="#1A1A1A",
            va="bottom",
            ha="left",
        )

        ax_a.set_title("(b) After projected Steiner split (0 neg.)", fontsize=10.0)
        ax_a.set_xlabel("x [m]")
        ax_a.set_ylabel("y [m]")
        ax_a.set_xlim(xmin, xmax)
        ax_a.set_ylim(ymin, ymax)
        ax_a.set_aspect("equal", adjustable="box")
        figstyle.spectrum_grid(ax_a)
        figstyle.place_legend_below(ax_a, ncol=1, y_offset=-0.20, fontsize=8.6)

        figstyle.save(fig, out_path, close=True, auto_legend_below=False)

    return out_path


def plot_dt_ratios(data: dict, out_path: Path) -> Path:
    """Render Figure (b): negative-face count and ``dt_max`` per repair path."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cases = data.get("cases", [])
    if not cases:
        raise ValueError("No cases in steiner.json")
    sc = cases[0]

    v_rf = sc["variant_i_repair_false"]
    v_w3 = sc["variant_ii_wave3_repair"]
    v_auto = sc["variant_iii_steiner_repair"]
    v_cad0 = sc.get("variant_iv_cad_projector", v_auto)
    v_cad6 = sc.get("variant_iv_cad_projector_sweeps6", v_auto)
    v_ana = sc.get("variant_v_analytic_projector", v_auto)

    path_rows = [
        ("repair=False", v_rf, "#D55E00"),
        ("Wave-3 flips\n(split=None)", v_w3, "#D55E00"),
        ("W4 auto\n(FeatureCurve)", v_auto, "#0072B2"),
        ("W4 CAD proj.\n(0 sweeps)", v_cad0, "#0072B2"),
        ("W4 CAD proj.\n(6 sweeps)", v_cad6, "#009E73"),
        ("Analytic\nprojector", v_ana, "#56B4E9"),
    ]

    suite = data.get("recorded_repair_suite", [])

    with figstyle.apply_style():
        fig, axes = figstyle.new_figure(
            width_in=7.6,
            height_in=4.15,
            ncols=2,
            nrows=1,
            gridspec_kw={"wspace": 0.48},
        )
        ax_paths, ax_suite = axes

        # ------------------------------------------------------------------
        # Panel (a): Cylinder-rim benchmark across the repair paths
        # ------------------------------------------------------------------
        y_pos = np.arange(len(path_rows), dtype=float)
        dt_ps = np.array([float(r[1]["dt_max"]) * 1e12 for r in path_rows], dtype=float)
        colors_p = [r[2] for r in path_rows]
        labels_p = [r[0] for r in path_rows]

        bars = ax_paths.barh(
            y_pos,
            dt_ps,
            height=0.62,
            color=colors_p,
            edgecolor="#1A1A1A",
            linewidth=0.7,
            zorder=3,
        )
        dt_ref_ps = float(v_rf["dt_max"]) * 1e12
        ax_paths.axvline(
            dt_ref_ps,
            color="#1A1A1A",
            linestyle="--",
            linewidth=1.2,
            zorder=2,
        )

        for bar, (_, ev, _) in zip(bars, path_rows):
            w = float(bar.get_width())
            neg = int(ev["n_negative_faces"])
            st_cnt = int(ev.get("steiner_nodes", 0))
            ratio = w / dt_ref_ps
            status = f"neg={neg} (st={st_cnt}), {w:.1f} ps ({ratio:.2f}×)"
            ax_paths.text(
                w + 0.55,
                bar.get_y() + 0.5 * bar.get_height(),
                status,
                va="center",
                ha="left",
                fontsize=8.5,
                color="#1A1A1A",
            )

        ax_paths.set_yticks(y_pos)
        ax_paths.set_yticklabels(labels_p)
        ax_paths.invert_yaxis()
        ax_paths.set_xlim(0.0, 56.0)
        ax_paths.set_xlabel("Max stable step Δt_max [ps]")
        ax_paths.set_title("(a) Cylinder-rim repair paths", fontsize=10.0)
        figstyle.spectrum_grid(ax_paths)

        # Custom legend handles for Panel (a)
        ax_paths.plot(
            [],
            [],
            color="#D55E00",
            marker="s",
            linestyle="none",
            label="Unrepaired (neg = 1, guard fails)",
        )
        ax_paths.plot(
            [],
            [],
            color="#0072B2",
            marker="s",
            linestyle="none",
            label="Repaired (neg = 0, 0 sweeps)",
        )
        ax_paths.plot(
            [],
            [],
            color="#009E73",
            marker="s",
            linestyle="none",
            label="Repaired + 6 CAD sliding sweeps",
        )
        figstyle.place_legend_below(ax_paths, ncol=1, y_offset=-0.20, fontsize=8.6)

        # ------------------------------------------------------------------
        # Panel (b): 6-case CAD suite (pillbox, layered, muon; none & gmsh)
        # ------------------------------------------------------------------
        if suite:
            y_s = np.arange(len(suite), dtype=float)
            gains = np.array(
                [float(r["dt_gain_vs_repair_false"]) for r in suite], dtype=float
            )
            s_labels = [f"{r['case']}\n({r['optimize']})" for r in suite]
            s_colors = [
                "#0072B2" if r["optimize"] == "none" else "#009E73" for r in suite
            ]
            s_bars = ax_suite.barh(
                y_s,
                gains,
                height=0.62,
                color=s_colors,
                edgecolor="#1A1A1A",
                linewidth=0.7,
                zorder=3,
            )
            ax_suite.axvline(
                1.0,
                color="#1A1A1A",
                linestyle="--",
                linewidth=1.2,
                zorder=2,
            )
            for bar, r in zip(s_bars, suite):
                w = float(bar.get_width())
                b_neg = int(r["before_neg"])
                rf_neg = int(r["repair_false_neg"])
                rt_neg = int(r["repair_true_neg"])
                dt_rt_ps = float(r["dt_max_repair_true_s"]) * 1e12
                txt = f"neg {b_neg}→{rf_neg}→{rt_neg}, {dt_rt_ps:.1f} ps ({w:.3f}×)"
                ax_suite.text(
                    w + 0.02,
                    bar.get_y() + 0.5 * bar.get_height(),
                    txt,
                    va="center",
                    ha="left",
                    fontsize=8.5,
                    color="#1A1A1A",
                )
            ax_suite.set_yticks(y_s)
            ax_suite.set_yticklabels(s_labels)
            ax_suite.invert_yaxis()
            ax_suite.set_xlim(0.0, 2.25)
            ax_suite.set_xlabel("Δt_max(repaired) / Δt_max(repair=False) [-]")
            ax_suite.set_title(
                "(b) 6-case CAD suite (cpw=10)",
                fontsize=10.0,
            )
            figstyle.spectrum_grid(ax_suite)
            ax_suite.plot(
                [],
                [],
                color="#0072B2",
                marker="s",
                linestyle="none",
                label="Delaunay (optimize='none', neg → 0)",
            )
            ax_suite.plot(
                [],
                [],
                color="#009E73",
                marker="s",
                linestyle="none",
                label="Gmsh 3D opt. (optimize='gmsh', neg → 0)",
            )
            ax_suite.plot(
                [],
                [],
                color="#1A1A1A",
                linestyle="--",
                marker="none",
                label="Parity line Δt ratio = 1.0",
            )
            figstyle.place_legend_below(ax_suite, ncol=1, y_offset=-0.20, fontsize=8.6)

        figstyle.save(fig, out_path, close=True, auto_legend_below=False)

    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json",
        type=Path,
        default=_DEFAULT_JSON,
        help="Input steiner.json path.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_DEFAULT_OUT_DIR,
        help="Output figure directory.",
    )
    args = parser.parse_args()

    raw = args.json.read_text(encoding="utf-8")
    if "END SUMMARY" in raw:
        raw = raw.split("END SUMMARY", 1)[1].lstrip()
    data = json.loads(raw)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    p1 = plot_rim_zoom(data, args.out_dir / "steiner_rim_zoom.png")
    print(f"Wrote {p1}")
    p2 = plot_dt_ratios(data, args.out_dir / "steiner_dt_ratios.png")
    print(f"Wrote {p2}")


if __name__ == "__main__":
    main()
