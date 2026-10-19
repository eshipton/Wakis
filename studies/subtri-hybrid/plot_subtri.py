# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Figures for the `subtri-hybrid` benchmark study (`studies/subtri-hybrid`):

a. ``figures/mesh_dimer_gap.png``: B3-g5 mesh at h = 2 nm (full patch and gap zoom).
b. ``figures/error_vs_work.png``: 3x3 log-log error vs work across all 9 cases.
c. ``figures/lts_gain.png``: work(subtri-hybrid-nolts) / work(subtri-hybrid) vs h
   for B2/B3 with chosen LTS step ratio p and fine fraction annotated.

Usage
-----
    python studies/subtri-hybrid/plot_subtri.py [--offline] [--data-dir DIR] [--fig-dir DIR]
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import compare as C  # noqa: E402

DATA = HERE / "data"
FIG = HERE / "figures"

plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 150, "font.size": 8.5})

ARM_STYLE = {
    # Ours (bold)
    "subtri-hybrid": dict(
        color="#D55E00", ls="-", lw=2.2, marker="o", ms=5.0, zorder=6
    ),
    "subtri-hybrid-nolts": dict(
        color="#D55E00", ls="--", lw=1.9, marker="s", ms=4.6, mfc="white", zorder=5
    ),
    "subtri-hybrid-r4": dict(
        color="#D55E00", ls="-", lw=2.2, marker="o", ms=5.0, zorder=6
    ),
    "subtri-hybrid-r2": dict(
        color="#E69F00", ls="--", lw=2.0, marker="s", ms=4.6, mfc="white", zorder=5
    ),
    # Baselines (thin grey/coloured lines with markers)
    "hybrid-quadtri": dict(
        color="#009E73", ls="-", lw=1.0, marker="P", ms=4.0, alpha=0.85
    ),
    "tri-graded": dict(color="#0072B2", ls="-", lw=1.0, marker="o", ms=3.6, alpha=0.8),
    "tri-graded-r2": dict(
        color="#0072B2", ls="--", lw=1.0, marker="o", ms=3.6, alpha=0.8
    ),
    "tri-graded-r4": dict(
        color="#0072B2", ls="-", lw=1.0, marker="^", ms=3.8, alpha=0.8
    ),
    "tri-graded-pml": dict(
        color="#56B4E9", ls="-", lw=1.0, marker="v", ms=3.6, alpha=0.8
    ),
    "tri-graded-pml-lts2": dict(
        color="#56B4E9", ls=":", lw=1.1, marker="d", ms=3.8, alpha=0.85
    ),
    "tri-graded-pml2-patch": dict(
        color="#56B4E9", ls="-.", lw=1.0, marker="<", ms=3.8, alpha=0.85
    ),
    "cart-conformal-all-graded": dict(
        color="#7F7F7F", ls="-", lw=1.0, marker="s", ms=3.5, alpha=0.8
    ),
    "cart-conformal-all-graded-r2": dict(
        color="#7F7F7F", ls="--", lw=1.0, marker="s", ms=3.5, alpha=0.8
    ),
    "cart-conformal-all-graded-r4": dict(
        color="#555555", ls="-", lw=1.0, marker="D", ms=3.5, alpha=0.8
    ),
    "nfit-ogrid": dict(color="#CC79A7", ls="-", lw=1.0, marker="h", ms=3.8, alpha=0.8),
    "nfit-boxed": dict(color="#CC79A7", ls="--", lw=1.0, marker="H", ms=3.8, alpha=0.8),
    "dgtd-p2": dict(color="#8C564B", ls="-.", lw=1.0, marker=">", ms=3.6, alpha=0.8),
    "dgtd-p3": dict(color="#9467BD", ls="-.", lw=1.0, marker="*", ms=4.2, alpha=0.8),
    "subgrid-2to1": dict(
        color="#333333", ls=":", lw=1.0, marker="X", ms=3.8, alpha=0.75
    ),
    "subgrid-2to1-gap4-lts8": dict(
        color="#333333", ls=":", lw=1.0, marker="X", ms=3.8, alpha=0.75
    ),
    "voronoi-poly": dict(
        color="#BCBD22", ls="-", lw=1.0, marker="p", ms=3.8, alpha=0.8
    ),
}

PRIMARY_METRIC = {
    "B1-TE": ("q_sca_rms_rel", (0.02, 0.01)),
    "B1-TM": ("q_sca_rms_rel", (0.02, 0.01)),
    "B2-Au-TE": ("q_ext_nrms", (0.02, 0.01, 0.005, 0.001)),
    "B2-Au-TM": ("q_ext_nrms", (0.02, 0.01, 0.005, 0.001)),
    "B2-Ag-TE": ("q_ext_nrms", (0.02, 0.01, 0.005, 0.001)),
    "B2-Ag-TM": ("q_ext_nrms", (0.02, 0.01, 0.005, 0.001)),
    "B3-g10": ("enh_peak", (0.05, 0.02, 0.01)),
    "B3-g5": ("enh_peak", (0.05, 0.02, 0.01)),
    "B3-g2": ("enh_peak", (0.05, 0.02, 0.01)),
}


def _add_scale_bar(ax, length_nm, xlim, ylim):
    """Draw a clean scale bar in the bottom-left corner of ``ax``."""
    span_x = xlim[1] - xlim[0]
    span_y = ylim[1] - ylim[0]
    x0 = xlim[0] + 0.06 * span_x
    x1 = x0 + length_nm
    y0 = ylim[0] + 0.07 * span_y
    ax.plot(
        [x0, x1],
        [y0, y0],
        color="k",
        lw=2.5,
        solid_capstyle="butt",
        zorder=8,
    )
    ax.text(
        0.5 * (x0 + x1),
        y0 + 0.025 * span_y,
        f"{length_nm:g} nm",
        ha="center",
        va="bottom",
        fontsize=7.5,
        fontweight="bold",
        color="k",
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8),
        zorder=8,
    )


def plot_mesh_dimer_gap(fig_dir=FIG):
    """
    Generate ``mesh_dimer_gap.png``: B3-g5 mesh at h = 2 nm with full patch
    (|x|, |y| < 60 nm) and gap zoom (|x|, |y| < 8 nm).
    """
    from wakis.benchmarks import get_case
    from wakis.benchmarks.arms.subtri2d import SubTriArm

    case = get_case("B3-g5")
    arm = SubTriArm()
    mesh, solver, info = arm.build(case, 2e-9)

    H = float(info["H"])
    max_level = int(info["levels"])
    nodes_nm = mesh.nodes * 1e9

    sq_polys = []
    sq_levels = []
    tri_polys = []
    for i in range(mesh.n_cells):
        c = mesh.cell(i)
        poly = nodes_nm[c]
        if len(c) == 4:
            side = np.sqrt(float(mesh.cell_area[i]))
            lvl = int(np.clip(np.round(np.log2(H / side)), 0, max_level))
            sq_polys.append(poly)
            sq_levels.append(lvl)
        else:
            tri_polys.append(poly)

    sq_levels = np.asarray(sq_levels, dtype=int)
    cmap = plt.get_cmap("viridis", max_level + 1)
    bounds = np.arange(-0.5, max_level + 1.5, 1.0)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    sq_colors = cmap(norm(sq_levels))

    hanging_nm = nodes_nm[solver._hanging]
    circles_nm = [
        (cy["center"][0] * 1e9, cy["center"][1] * 1e9, cy["radius"] * 1e9)
        for cy in case.cylinders
    ]

    fig, axs = plt.subplots(1, 2, figsize=(11.0, 4.8), constrained_layout=True)
    panels = [
        (axs[0], 60.0, 20.0, 2.2, "(a) Patch region (|x|, |y| < 60 nm)"),
        (axs[1], 8.0, 2.0, 3.5, "(b) Gap zoom (|x|, |y| < 8 nm)"),
    ]

    for ax, lim, bar_nm, h_ms, title in panels:
        # Filter cells visible within [-lim, lim] with margin
        margin = 10.0
        sq_vis = [
            (p, col)
            for p, col in zip(sq_polys, sq_colors)
            if p[:, 0].max() >= -lim - margin
            and p[:, 0].min() <= lim + margin
            and p[:, 1].max() >= -lim - margin
            and p[:, 1].min() <= lim + margin
        ]
        if sq_vis:
            pc_sq = PolyCollection(
                [p for p, _ in sq_vis],
                facecolors=[col for _, col in sq_vis],
                edgecolors="0.2",
                linewidths=0.35,
                alpha=0.85,
                zorder=1,
            )
            ax.add_collection(pc_sq)

        tri_vis = [
            p
            for p in tri_polys
            if p[:, 0].max() >= -lim - margin
            and p[:, 0].min() <= lim + margin
            and p[:, 1].max() >= -lim - margin
            and p[:, 1].min() <= lim + margin
        ]
        if tri_vis:
            pc_tri = PolyCollection(
                tri_vis,
                facecolors="#DCDCDC",
                edgecolors="0.3",
                linewidths=0.35,
                zorder=2,
            )
            ax.add_collection(pc_tri)

        for cx, cy, r in circles_nm:
            circ = plt.Circle(
                (cx, cy), r, fill=False, edgecolor="#D62728", lw=1.4, zorder=4
            )
            ax.add_patch(circ)

        h_mask = (np.abs(hanging_nm[:, 0]) <= lim) & (np.abs(hanging_nm[:, 1]) <= lim)
        if np.any(h_mask):
            ax.plot(
                hanging_nm[h_mask, 0],
                hanging_nm[h_mask, 1],
                "ko",
                ms=h_ms,
                label="hanging nodes",
                zorder=5,
            )

        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_xlabel("x [nm]")
        ax.set_ylabel("y [nm]")
        ax.set_title(title)
        _add_scale_bar(ax, bar_nm, (-lim, lim), (-lim, lim))

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(
        sm,
        ax=axs,
        ticks=list(range(max_level + 1)),
        label="level",
        shrink=0.82,
        pad=0.02,
    )

    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_path = fig_dir / "mesh_dimer_gap.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_error_vs_work(data_dir=DATA, fig_dir=FIG, offline=False):
    """
    Generate ``error_vs_work.png``: 3x3 log-log error vs work across all 9 cases.
    """
    local_df = C.load_local_df(data_dir)
    cases = list(C.CASE_SPECS.keys())
    fig, axs = plt.subplots(3, 3, figsize=(13.5, 11.0))

    legend_handles = {}

    for ax, case in zip(axs.ravel(), cases):
        spec = C.CASE_SPECS[case]
        metric, targets = PRIMARY_METRIC[case]
        mcol = f"m.{metric}"
        df = C.load_case_df(case, data_dir=data_dir, offline=offline, local_df=local_df)

        all_arms = list(spec["ours"]) + list(spec["baselines"])
        for arm in all_arms:
            if df.empty or "arm" not in df.columns or mcol not in df.columns:
                continue
            g = df[df["arm"] == arm].copy()
            if g.empty:
                continue
            g["work"] = pd.to_numeric(g["work"], errors="coerce")
            g[mcol] = pd.to_numeric(g[mcol], errors="coerce").abs()
            # Filter out non-finite or blown-up (> 5.0 relative error) points
            g = g[
                np.isfinite(g["work"])
                & (g["work"] > 0)
                & np.isfinite(g[mcol])
                & (g[mcol] > 0)
                & (g[mcol] <= 5.0)
            ]
            if g.empty:
                continue
            g = g.sort_values("work")
            st = dict(
                ARM_STYLE.get(
                    arm,
                    dict(color="#666666", ls="-", lw=1.0, marker="o", ms=3.5),
                )
            )
            (line,) = ax.loglog(
                g["work"].to_numpy(float),
                g[mcol].to_numpy(float),
                label=arm,
                **st,
            )
            if arm not in legend_handles:
                legend_handles[arm] = line

        for t in targets:
            ax.axhline(t, color="0.45", ls=":", lw=0.9, zorder=1)

        ax.set_title(f"{case} ({metric})")
        ax.set_xlabel("work (unknowns × steps)")
        ax.set_ylabel(metric)
        ax.grid(True, which="both", alpha=0.22)

    if legend_handles:
        # Order legend with our arms first
        our_names = [
            a
            for a in (
                "subtri-hybrid",
                "subtri-hybrid-nolts",
                "subtri-hybrid-r4",
                "subtri-hybrid-r2",
            )
            if a in legend_handles
        ]
        other_names = [a for a in legend_handles if a not in our_names]
        ordered = our_names + other_names
        fig.legend(
            [legend_handles[a] for a in ordered],
            ordered,
            loc="lower center",
            ncol=min(6, max(1, len(ordered))),
            fontsize=7.5,
            frameon=True,
            bbox_to_anchor=(0.5, 0.0),
        )
        fig.tight_layout(rect=(0.0, 0.07, 1.0, 1.0))
    else:
        fig.tight_layout()

    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_path = fig_dir / "error_vs_work.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def _dedup_raw_records(records):
    """Deduplicate raw JSON records by ``(case, arm, h_nm)`` keeping latest ``published``."""
    valid = [
        r for r in records if not r.get("error") and "cost" in r and "resolution" in r
    ]
    valid.sort(key=lambda r: str(r.get("published") or ""))
    by_key = {}
    for r in valid:
        key = (r["case"], r["arm"], round(float(r["resolution"]) * 1e9, 6))
        by_key[key] = r
    return by_key


def _work_of_rec(r):
    c = r["cost"]
    if c.get("work") is not None:
        return float(c["work"])
    return float(c["unknowns"]) * float(c["steps"])


def plot_lts_gain(data_dir=DATA, fig_dir=FIG):
    """
    Generate ``lts_gain.png``: for B2/B3, ``work(subtri-hybrid-nolts) / work(subtri-hybrid)``
    at equal ``h`` vs ``h``, with chosen ``p`` and fine fraction annotated.
    """
    recs = C.load_local_records(data_dir)
    by_key = _dedup_raw_records(recs)

    b2_cases = ["B2-Au-TE", "B2-Au-TM", "B2-Ag-TE", "B2-Ag-TM"]
    b3_cases = ["B3-g10", "B3-g5", "B3-g2"]

    fig, axs = plt.subplots(1, 2, figsize=(11.0, 4.4), sharey=True)
    groups = [
        (axs[0], b2_cases, "(a) B2 plasmonic cylinder"),
        (axs[1], b3_cases, "(b) B3 plasmonic dimer"),
    ]
    markers = ("o", "s", "^", "D")

    for ax, group_cases, title in groups:
        has_points = False
        seen_annots = set()
        all_h = []
        for idx, case in enumerate(group_cases):
            # Collect resolutions where subtri-hybrid exists
            h_keys = sorted(
                {k[2] for k in by_key if k[0] == case and k[1] == "subtri-hybrid"}
            )
            hs_nm, gains, annots = [], [], []
            for h_nm in h_keys:
                r_lts = by_key.get((case, "subtri-hybrid", h_nm))
                r_nolts = by_key.get((case, "subtri-hybrid-nolts", h_nm))
                if r_lts is None or r_nolts is None:
                    continue
                w_lts = _work_of_rec(r_lts)
                w_nolts = _work_of_rec(r_nolts)
                if w_lts <= 0:
                    continue
                gain = w_nolts / w_lts
                lts_info = r_lts["cost"].get("lts")
                if isinstance(lts_info, dict):
                    p = int(lts_info.get("p", 1))
                    ff = float(lts_info.get("fine_fraction", 0.0))
                else:
                    p = 1
                    ff = 0.0
                hs_nm.append(float(r_lts["resolution"]) * 1e9)
                gains.append(gain)
                annots.append((p, ff))

            if not hs_nm:
                continue
            has_points = True
            all_h.extend(hs_nm)
            col = f"C{idx}"
            mk = markers[idx % len(markers)]
            ax.plot(
                hs_nm,
                gains,
                marker=mk,
                color=col,
                lw=1.6,
                ms=5.0,
                label=case,
            )
            for j, (x, y, (p, ff)) in enumerate(zip(hs_nm, gains, annots)):
                akey = (round(x, 3), round(y, 2), p, round(ff * 100))
                if akey in seen_annots:
                    continue
                seen_annots.add(akey)
                dy = 7 if (j % 2 == 0) else -12
                ax.annotate(
                    f"p={p}, {ff * 100:.0f}%",
                    xy=(x, y),
                    xytext=(0, dy),
                    textcoords="offset points",
                    ha="center",
                    fontsize=7.5,
                    color="0.15",
                    bbox=dict(
                        boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.75
                    ),
                )

        ax.axhline(1.0, color="0.5", ls="--", lw=0.9)
        ax.set_xscale("log")
        if all_h:
            ticks = sorted({round(v, 3) for v in all_h})
            ax.set_xticks(ticks)
            ax.set_xticklabels([f"{t:g}" for t in ticks])
            ax.minorticks_off()
        ax.margins(x=0.12, y=0.18)
        ax.set_xlabel("h [nm]")
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.25)
        if has_points:
            ax.legend(fontsize=7.5)

    axs[0].set_ylabel("work(subtri-hybrid-nolts) / work(subtri-hybrid)")
    fig.tight_layout()

    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_path = fig_dir / "lts_gain.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--offline",
        action="store_true",
        help="Use cached data/baselines_<case>.csv without network access.",
    )
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=DATA,
        help="Directory containing local JSON records and baseline CSVs.",
    )
    ap.add_argument(
        "--fig-dir",
        type=Path,
        default=FIG,
        help="Output directory for generated PNG figures.",
    )
    ap.add_argument(
        "--no-mesh",
        action="store_true",
        help="Skip building the B3-g5 mesh figure.",
    )
    args = ap.parse_args(argv)

    if not args.no_mesh:
        plot_mesh_dimer_gap(fig_dir=args.fig_dir)
    plot_error_vs_work(
        data_dir=args.data_dir, fig_dir=args.fig_dir, offline=args.offline
    )
    plot_lts_gain(data_dir=args.data_dir, fig_dir=args.fig_dir)


if __name__ == "__main__":
    main()
