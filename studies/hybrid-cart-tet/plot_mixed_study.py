# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Plot the PEC-sphere equal-accuracy and cost-per-unknown-step study from
`studies/hybrid-cart-tet/results_mixed_study.json` (written by
`studies/hybrid-cart-tet/run_mixed_study.py`).

Writes two PNGs under `studies/hybrid-cart-tet/figures/`:
* `fig_hybrid_mixed_tts_vs_error.png`: time-to-solution [s] for one physical
  source period `T_phys = lambda / c0` vs far-zone relative L2 error [-];
* `fig_hybrid_mixed_cost_per_unknown_step.png`: cost per unknown-step
  [s / (unknown * step)] bars for each variant and the `SolverFIT3D`
  Cartesian reference at each `N`.

Usage:
    python studies/hybrid-cart-tet/plot_mixed_study.py
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figures")
DEFAULT_JSON = os.path.join(HERE, "results_mixed_study.json")

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "savefig.dpi": 150,
        "figure.dpi": 100,
    }
)

VARIANT_STYLE = {
    "consistent": {
        "label": "hybrid, consistent mass",
        "color": "C0",
        "marker": "o",
        "ls": "-",
    },
    "mixed": {
        "label": "hybrid, mixed mass",
        "color": "C1",
        "marker": "s",
        "ls": "--",
    },
    "mixed_smoothed": {
        "label": "hybrid, mixed mass + smooth_patch",
        "color": "C2",
        "marker": "^",
        "ls": "-",
    },
    "consistent_smoothed": {
        "label": "hybrid, consistent mass + smooth_patch",
        "color": "C4",
        "marker": "v",
        "ls": "-.",
    },
    "staircase": {
        "label": "Cartesian staircase (HybridFIT3D)",
        "color": "0.25",
        "marker": "D",
        "ls": ":",
    },
    "tet": {
        "label": "all-tetrahedral physical region",
        "color": "C3",
        "marker": "P",
        "ls": "-.",
    },
}


def plot_tts_vs_error(data, out_path):
    table = data["table"]
    t_phys = data.get("T_phys", 0.0)
    variants = []
    for row in table:
        v = row["variant"]
        if v not in variants:
            variants.append(v)

    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    for v in variants:
        rows = sorted([r for r in table if r["variant"] == v], key=lambda r: r["N"])
        if not rows:
            continue
        st = VARIANT_STYLE.get(v, {"label": v, "color": "C5", "marker": "o", "ls": "-"})
        errs = [r["far_zone_error"] for r in rows]
        tts = [r["time_to_solution_s"] for r in rows]
        ax.loglog(
            errs,
            tts,
            marker=st["marker"],
            linestyle=st["ls"],
            color=st["color"],
            label=st["label"],
            ms=6,
            lw=1.6,
        )
        for r in rows:
            ax.annotate(
                f"N={r['N']}",
                (r["far_zone_error"], r["time_to_solution_s"]),
                textcoords="offset points",
                xytext=(5, 4),
                fontsize=8,
                color=st["color"],
            )

    ax.set_xlabel(
        "Far-zone relative $L_2$ error "
        "$\\|\\mathbf{e}_{\\mathrm{num}} - \\mathbf{e}_{\\mathrm{ref}}\\|_2"
        " / \\|\\mathbf{e}_{\\mathrm{ref}}\\|_2$ [-]"
    )
    ax.set_ylabel(
        f"Time to solution for $T_{{\\mathrm{{phys}}}} = \\lambda / c_0"
        f" = {t_phys:.3e}$ s [s]"
    )
    ax.set_title(
        "PEC sphere ($ka = 2$): time-to-solution vs far-zone Mie error",
        fontsize=10,
    )
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_cost_per_unknown_step(data, out_path):
    table = data["table"]
    ns = sorted({int(r["N"]) for r in table})
    variants = []
    for row in table:
        v = row["variant"]
        if v not in variants:
            variants.append(v)

    bar_keys = variants + ["SolverFIT3D_cart"]
    n_bars = len(bar_keys)
    x = np.arange(len(ns), dtype=float)
    width = 0.8 / max(n_bars, 1)

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    for idx, key in enumerate(bar_keys):
        offset = (idx - 0.5 * (n_bars - 1)) * width
        vals = []
        for N in ns:
            if key == "SolverFIT3D_cart":
                cart_n = data.get("cartesian_solverfit3d", {}).get(f"N={N}")
                if cart_n is not None:
                    vals.append(cart_n["cart_cost_per_unknown_step_s"])
                else:
                    matches = [r for r in table if int(r["N"]) == N]
                    vals.append(
                        matches[0]["cart_cost_per_unknown_step_s"]
                        if matches
                        else np.nan
                    )
            else:
                matches = [r for r in table if int(r["N"]) == N and r["variant"] == key]
                vals.append(
                    matches[0]["cost_per_unknown_step_s"] if matches else np.nan
                )
        if key == "SolverFIT3D_cart":
            label = "SolverFIT3D Cartesian reference (same h)"
            color = "0.55"
        else:
            st = VARIANT_STYLE.get(key, {"label": key, "color": f"C{idx}"})
            label = st["label"]
            color = st["color"]
        ax.bar(x + offset, vals, width=width * 0.92, label=label, color=color)

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"N = {N} (h = 1/{N} m)" for N in ns])
    ax.set_xlabel("Cartesian grid resolution $N$ [cells / wavelength]")
    ax.set_ylabel("Cost per unknown-step [s / (unknown $\\cdot$ step)]")
    ax.set_title(
        "PEC sphere time-domain benchmark (200 steps, 1 core): cost per unknown-step",
        fontsize=10,
    )
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        default=DEFAULT_JSON,
        help="Path to results_mixed_study.json",
    )
    ap.add_argument(
        "--out-dir",
        default=FIG_DIR,
        help="Directory to write PNG figures",
    )
    args = ap.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    os.makedirs(args.out_dir, exist_ok=True)
    p1 = os.path.join(args.out_dir, "fig_hybrid_mixed_tts_vs_error.png")
    p2 = os.path.join(args.out_dir, "fig_hybrid_mixed_cost_per_unknown_step.png")
    plot_tts_vs_error(data, p1)
    plot_cost_per_unknown_step(data, p2)
    print(f"Wrote {p1} and {p2}", flush=True)


if __name__ == "__main__":
    main()
