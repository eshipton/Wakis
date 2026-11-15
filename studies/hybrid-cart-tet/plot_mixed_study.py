# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Plot the PEC-sphere equal-accuracy and cost-per-unknown-step study from
`studies/hybrid-cart-tet/results_mixed_study.json` and
`studies/hybrid-cart-tet/results_mixed_study_N20.json` (written by
`studies/hybrid-cart-tet/run_mixed_study.py` and merged by key `table`).

Writes two PNGs under `studies/hybrid-cart-tet/figures/`:
* `fig_hybrid_mixed_tts_vs_error.png`: one-period time to solution [s] for one
  physical source period `T_phys = lambda / c0` vs far-zone relative L2 error
  [dimensionless] for each `(N, variant)`;
* `fig_hybrid_mixed_cost_per_unknown_step.png`: cost per unknown-step
  [s / (unknown * step)] vs unknowns `n_unknowns` [dimensionless] (log-log) for
  each `(N, variant)` with the `SolverFIT3D` Cartesian reference line.

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
DEFAULT_JSON_N20 = os.path.join(HERE, "results_mixed_study_N20.json")
DEFAULT_JSONS = (DEFAULT_JSON, DEFAULT_JSON_N20)

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


def merge_study_data(datasets, mutate=None):
    """Merge study JSON dictionaries by key ``table``."""
    if mutate not in (None, "drop_extra"):
        raise ValueError(f"Unsupported mutate={mutate!r}")
    if not datasets:
        raise ValueError("No input datasets provided")
    merged = dict(datasets[0])
    use_datasets = datasets[:1] if mutate == "drop_extra" else datasets
    by_key = {}
    variant_order = []
    cart_refs = {}
    results_by_n = {}
    for d in use_datasets:
        cart_refs.update(d.get("cartesian_solverfit3d", {}))
        results_by_n.update(d.get("results", {}))
        for row in d.get("table", []):
            v = row["variant"]
            if v not in variant_order:
                variant_order.append(v)
            by_key[(int(row["N"]), v)] = row
    merged["cartesian_solverfit3d"] = cart_refs
    merged["results"] = results_by_n
    ns = sorted({k[0] for k in by_key})
    merged["table"] = [
        by_key[(N, v)] for N in ns for v in variant_order if (N, v) in by_key
    ]
    return merged


def load_merged_study(paths=None, mutate=None):
    """Load and merge ``results_mixed_study.json`` and ``results_mixed_study_N20.json``."""
    if paths is None:
        paths = [p for p in DEFAULT_JSONS if os.path.exists(p)]
    elif isinstance(paths, str):
        paths = [paths]
        if os.path.abspath(paths[0]) == os.path.abspath(
            DEFAULT_JSON
        ) and os.path.exists(DEFAULT_JSON_N20):
            paths.append(DEFAULT_JSON_N20)
    datasets = []
    for p in paths:
        with open(p) as f:
            datasets.append(json.load(f))
    return merge_study_data(datasets, mutate=mutate)


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
        rows = sorted(
            [
                r
                for r in table
                if r["variant"] == v and r.get("time_to_solution_s") is not None
            ],
            key=lambda r: r["N"],
        )
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
        " / \\|\\mathbf{e}_{\\mathrm{ref}}\\|_2$ [dimensionless]"
    )
    ax.set_ylabel(
        f"One-period time to solution ($T_{{\\mathrm{{phys}}}} = \\lambda / c_0"
        f" = {t_phys:.3e}$ s) [s]"
    )
    ax.set_title(
        "PEC sphere ($ka = 2$): one-period time to solution vs far-zone Mie error",
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

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    all_unk = []
    for idx, v in enumerate(variants):
        rows = sorted(
            [
                r
                for r in table
                if r["variant"] == v and r.get("cost_per_unknown_step_s") is not None
            ],
            key=lambda r: r["N"],
        )
        if not rows:
            continue
        st = VARIANT_STYLE.get(
            v, {"label": v, "color": f"C{idx}", "marker": "o", "ls": "-"}
        )
        unk = [r["n_unknowns"] for r in rows]
        cost = [r["cost_per_unknown_step_s"] for r in rows]
        all_unk.extend(unk)
        ax.loglog(
            unk,
            cost,
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
                (r["n_unknowns"], r["cost_per_unknown_step_s"]),
                textcoords="offset points",
                xytext=(5, 4),
                fontsize=8,
                color=st["color"],
            )

    cart_pts = []
    for N in ns:
        cart_n = data.get("cartesian_solverfit3d", {}).get(f"N={N}")
        if cart_n is not None and "cart_n_unknowns" in cart_n:
            cart_pts.append(
                (N, cart_n["cart_n_unknowns"], cart_n["cart_cost_per_unknown_step_s"])
            )
        else:
            matches = [r for r in table if int(r["N"]) == N]
            if matches:
                u = matches[0].get("cart_n_unknowns", matches[0]["n_unknowns"])
                c = matches[0]["cart_cost_per_unknown_step_s"]
                cart_pts.append((N, u, c))

    if cart_pts:
        cart_unk = [p[1] for p in cart_pts]
        cart_cost = [p[2] for p in cart_pts]
        all_unk.extend(cart_unk)
        u_min = min(all_unk)
        u_max = max(all_unk)
        ref_val = float(np.mean(cart_cost))
        ax.loglog(
            [u_min, u_max],
            [ref_val, ref_val],
            linestyle="--",
            color="0.45",
            lw=1.3,
            label="SolverFIT3D Cartesian reference line (same h)",
        )
        ax.loglog(
            cart_unk,
            cart_cost,
            marker="x",
            linestyle="-",
            color="0.35",
            ms=6,
            lw=1.2,
        )
        for N, u, c in cart_pts:
            ax.annotate(
                f"N={N}",
                (u, c),
                textcoords="offset points",
                xytext=(5, -10),
                fontsize=8,
                color="0.35",
            )

    ax.set_xlabel(
        "Number of unknowns $n_{\\mathrm{unknowns}} = n_{\\mathrm{free}} + n_{\\mathrm{faces}}$ [dimensionless]"
    )
    ax.set_ylabel("Cost per unknown-step [s / (unknown $\\cdot$ step)]")
    ax.set_title(
        "PEC sphere time-domain benchmark (200 steps, 1 core): cost per unknown-step vs unknowns",
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
        nargs="*",
        default=None,
        help=(
            "Paths to study JSON files (default: results_mixed_study.json and "
            "results_mixed_study_N20.json merged by key 'table')."
        ),
    )
    ap.add_argument(
        "--out-dir",
        default=FIG_DIR,
        help="Directory to write PNG figures",
    )
    args = ap.parse_args()

    data = load_merged_study(args.input)

    os.makedirs(args.out_dir, exist_ok=True)
    p1 = os.path.join(args.out_dir, "fig_hybrid_mixed_tts_vs_error.png")
    p2 = os.path.join(args.out_dir, "fig_hybrid_mixed_cost_per_unknown_step.png")
    plot_tts_vs_error(data, p1)
    plot_cost_per_unknown_step(data, p2)
    print(f"Wrote {p1} and {p2}", flush=True)


if __name__ == "__main__":
    main()
