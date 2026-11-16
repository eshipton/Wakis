r"""
Plot projected Steiner mesh-repair figures from `results/steiner.json`.

Usage:
    python studies/unstructured-cadmesh/plot_steiner.py
    python studies/unstructured-cadmesh/plot_steiner.py --in /tmp/w4_steiner_quick.json --figdir /tmp

Writes two figures to `studies/unstructured-cadmesh/figures/` (or `--figdir`):
1. `steiner_rim_zoom.png`: labelled 2D $(x, y)$ zoom of one stuck non-Delaunay
   face next to the rim circle before and after the projected rim-edge split,
   showing the incident tetrahedra, the analytic CAD circle $r = a$, the chord
   midpoint, and the projected Steiner node on the CAD circle.
2. `steiner_dt_ratios.png`: $\Delta t_{\max}$ ratios of projected Steiner repair
   (`split_project="auto"`) vs node relocation alone (`repair=False`) and vs
   wave-3 flip repair (`split_project=None`), annotated with solver guard
   negative-face counts before and after.
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RES = os.path.join(HERE, "results", "steiner.json")
DEFAULT_FIG = os.path.join(HERE, "figures")

plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3})


def _load_data(res_path):
    with open(res_path) as f:
        return json.load(f)


def _label(rec):
    opt = rec.get("optimize", "none")
    case = rec.get("case", "case")
    return f"{case}\n({opt})" if opt != "none" else f"{case}\n(Delaunay)"


def plot_rim_zoom(data, fig_dir):
    """
    Plot 2-panel (x, y) zoom of a stuck rim-adjacent face before and after
    splitting the encroached locked rim edge onto the CAD circle r = a.
    """
    zoom = data.get("zoom_example")
    if not zoom:
        return

    R = float(zoom.get("radius_m", 0.1))
    edge_pts = np.asarray(zoom["split_edge_xyz_m"], dtype=float)
    chord_mid = np.asarray(zoom["chord_midpoint_xyz_m"], dtype=float)
    steiner_pts = np.asarray(
        zoom.get("steiner_nodes_xyz_m") or [chord_mid], dtype=float
    )
    p_steiner = steiner_pts[0]
    face_pts = np.asarray(zoom["stuck_face_xyz_m"], dtype=float)
    tets_before = [
        np.asarray(t, dtype=float) for t in zoom.get("tets_before_xyz_m", [])
    ]
    tets_after = [np.asarray(t, dtype=float) for t in zoom.get("tets_after_xyz_m", [])]

    # Angular range around the split edge for drawing the analytic CAD circle arc
    th_a = float(np.arctan2(edge_pts[0, 1], edge_pts[0, 0]))
    th_b = float(np.arctan2(edge_pts[1, 1], edge_pts[1, 0]))
    if th_b < th_a:
        th_b += 2.0 * np.pi
    th_pad = 0.25 * max(abs(th_b - th_a), 0.2)
    theta = np.linspace(th_a - th_pad, th_b + th_pad, 200)
    arc_xy = np.column_stack([R * np.cos(theta), R * np.sin(theta)])

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))

    # Panel (a): Before split (2 incident tets + non-Delaunay face + encroached rim chord)
    ax0 = axes[0]
    for idx, tet in enumerate(tets_before):
        for i in range(4):
            for j in range(i + 1, 4):
                ax0.plot(
                    [tet[i, 0], tet[j, 0]],
                    [tet[i, 1], tet[j, 1]],
                    color="#7F8C8D",
                    linewidth=1.0,
                    alpha=0.75,
                    label="Incident tet edges"
                    if (idx == 0 and i == 0 and j == 1)
                    else None,
                )
    face_loop = np.vstack([face_pts, face_pts[:1]])
    ax0.fill(
        face_loop[:, 0],
        face_loop[:, 1],
        color="#E74C3C",
        alpha=0.22,
        label="Stuck face (H < 0)",
    )
    ax0.plot(
        arc_xy[:, 0],
        arc_xy[:, 1],
        color="#1F77B4",
        linestyle="--",
        linewidth=1.6,
        label="CAD rim circle (r = a)",
    )
    ax0.plot(
        edge_pts[:, 0],
        edge_pts[:, 1],
        color="#C0392B",
        linewidth=2.2,
        marker="o",
        markersize=5,
        label="Locked rim chord (a, b)",
    )
    ax0.scatter(
        [chord_mid[0]],
        [chord_mid[1]],
        color="#D35400",
        marker="x",
        s=55,
        zorder=5,
        label="Chord midpoint m (off CAD)",
    )
    ax0.set_title("(a) Before split: encroached locked rim edge", fontsize=9)
    ax0.set_xlabel("x [m]")
    ax0.set_ylabel("y [m]")
    ax0.set_aspect("equal", adjustable="datalim")
    ax0.legend(fontsize=7.5, loc="best")

    # Panel (b): After projected rim-edge split (4 child tets + Steiner node on r = a)
    ax1 = axes[1]
    for idx, tet in enumerate(tets_after):
        for i in range(4):
            for j in range(i + 1, 4):
                ax1.plot(
                    [tet[i, 0], tet[j, 0]],
                    [tet[i, 1], tet[j, 1]],
                    color="#27AE60",
                    linewidth=1.0,
                    alpha=0.75,
                    label="Split child tet edges"
                    if (idx == 0 and i == 0 and j == 1)
                    else None,
                )
    ax1.plot(
        arc_xy[:, 0],
        arc_xy[:, 1],
        color="#1F77B4",
        linestyle="--",
        linewidth=1.6,
        label="CAD rim circle (r = a)",
    )
    ax1.plot(
        [edge_pts[0, 0], p_steiner[0], edge_pts[1, 0]],
        [edge_pts[0, 1], p_steiner[1], edge_pts[1, 1]],
        color="#2ECC71",
        linewidth=2.2,
        marker="o",
        markersize=5,
        label="Sub-edges (a, p), (p, b)",
    )
    ax1.scatter(
        [chord_mid[0]],
        [chord_mid[1]],
        color="#D35400",
        marker="x",
        s=55,
        zorder=5,
        label="Chord midpoint m",
    )
    ax1.scatter(
        [p_steiner[0]],
        [p_steiner[1]],
        color="#117A65",
        marker="*",
        s=95,
        zorder=6,
        label="Projected Steiner node p (on r = a)",
    )
    ax1.set_title("(b) After projected split: Delaunay restored (H >= 0)", fontsize=9)
    ax1.set_xlabel("x [m]")
    ax1.set_ylabel("y [m]")
    ax1.set_aspect("equal", adjustable="datalim")
    ax1.legend(fontsize=7.5, loc="best")

    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "steiner_rim_zoom.png"), dpi=170)
    plt.close(fig)


def plot_dt_ratios(data, fig_dir):
    """
    Plot dt_max ratios of projected Steiner repair vs node relocation alone
    (repair=False) and vs wave-3 flip repair (split_project=None), with solver
    guard negative-face counts annotated.
    """
    records = data.get("all_runs") or data.get("cases", [])
    if not records:
        return

    labels = [_label(r) for r in records]
    x = np.arange(len(records))
    w = 0.34

    ratio_vs_reloc = [float(r["dt_gain_vs_repair_false"]) for r in records]
    ratio_vs_wave3 = [float(r["dt_gain_vs_wave3_repair"]) for r in records]

    neg_before = [int(r["before"]["n_negative_faces"]) for r in records]
    neg_reloc = [int(r["variant_i_repair_false"]["n_negative_faces"]) for r in records]
    neg_wave3 = [
        int(
            r.get("pure_wave3_repair", r["variant_ii_wave3_repair"])["n_negative_faces"]
        )
        for r in records
    ]
    neg_steiner = [
        int(r["variant_iii_steiner_repair"]["n_negative_faces"]) for r in records
    ]

    fig, ax = plt.subplots(1, 2, figsize=(10.8, 4.2))

    # Panel (a): dt_max ratios
    b1 = ax[0].bar(
        x - 0.5 * w,
        ratio_vs_reloc,
        width=w,
        color="#4C72B0",
        label=r"$\Delta t_{\max}$ (Steiner / repair=False)",
    )
    b2 = ax[0].bar(
        x + 0.5 * w,
        ratio_vs_wave3,
        width=w,
        color="#55A868",
        label=r"$\Delta t_{\max}$ (Steiner / wave-3 split_project=None)",
    )
    ax[0].axhline(1.0, color="#333333", linestyle="--", linewidth=0.8)
    for bar, n_w3, n_st in zip(b2, neg_wave3, neg_steiner):
        ax[0].annotate(
            f"neg:{n_w3}->{n_st}",
            (bar.get_x() + bar.get_width() / 2.0, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    for bar, n_rel in zip(b1, neg_reloc):
        ax[0].annotate(
            f"reloc:{n_rel}",
            (bar.get_x() + bar.get_width() / 2.0, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(labels, fontsize=8)
    ax[0].set_xlabel("CAD geometry (cpw = 10) [-]")
    ax[0].set_ylabel(
        r"Time-step ratio $\Delta t_{\max} / \Delta t_{\max,\mathrm{ref}}$ [-]"
    )
    ax[0].set_title(
        r"(a) $\Delta t_{\max}$ ratios with guard negative counts", fontsize=9
    )
    ax[0].legend(fontsize=7.5, loc="best")

    # Panel (b): Solver guard negative-face counts across variants
    w_neg = 0.20
    ax[1].bar(
        x - 1.5 * w_neg,
        neg_before,
        width=w_neg,
        color="#4C72B0",
        label="Before improve_dt",
    )
    ax[1].bar(
        x - 0.5 * w_neg,
        neg_reloc,
        width=w_neg,
        color="#DD8452",
        label="improve_dt(repair=False)",
    )
    ax[1].bar(
        x + 0.5 * w_neg,
        neg_wave3,
        width=w_neg,
        color="#C44E52",
        label="Wave-3 repair (split_project=None)",
    )
    ax[1].bar(
        x + 1.5 * w_neg,
        neg_steiner,
        width=w_neg,
        color="#55A868",
        label="Projected Steiner (split_project='auto')",
    )
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(labels, fontsize=8)
    ax[1].set_xlabel("CAD geometry (cpw = 10) [-]")
    ax[1].set_ylabel("Solver guard negative faces [count]")
    ax[1].set_title("(b) Non-Delaunay guard count before and after repair", fontsize=9)
    ax[1].legend(fontsize=7.5, loc="best")

    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "steiner_dt_ratios.png"), dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--in",
        dest="in_path",
        type=str,
        default=DEFAULT_RES,
        help="Input JSON path produced by steiner_study.py.",
    )
    parser.add_argument(
        "--figdir",
        type=str,
        default=DEFAULT_FIG,
        help="Output directory for generated PNG figures.",
    )
    args = parser.parse_args()

    os.makedirs(args.figdir, exist_ok=True)
    data = _load_data(args.in_path)
    plot_rim_zoom(data, args.figdir)
    plot_dt_ratios(data, args.figdir)


if __name__ == "__main__":
    main()
