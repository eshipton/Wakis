# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Plot the PEC-sphere equal-accuracy and cost-per-unknown-step study from
`studies/hybrid-cart-tet/results_mixed_study.json`,
`studies/hybrid-cart-tet/results_mixed_study_N20.json`, and
`studies/hybrid-cart-tet/results_mixed_study_N24.json` (written by
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
DEFAULT_JSON_N24 = os.path.join(HERE, "results_mixed_study_N24.json")
DEFAULT_JSONS = (DEFAULT_JSON, DEFAULT_JSON_N20, DEFAULT_JSON_N24)

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
        "ms": 7.0,
        "mfc": "white",
        "mew": 1.8,
        "zorder": 4,
    },
    "mixed": {
        "label": "hybrid, mixed mass",
        "color": "C1",
        "marker": "s",
        "ls": "--",
        "ms": 6.0,
        "mfc": "C1",
        "mew": 1.2,
        "zorder": 3,
    },
    "mixed_smoothed": {
        "label": "hybrid, mixed mass + smooth_patch",
        "color": "C2",
        "marker": "^",
        "ls": "-",
        "ms": 6.2,
        "mfc": "C2",
        "mew": 1.2,
        "zorder": 5,
    },
    "consistent_smoothed": {
        "label": "hybrid, consistent mass + smooth_patch",
        "color": "C4",
        "marker": "v",
        "ls": "-.",
        "ms": 6.0,
        "mfc": "C4",
        "mew": 1.2,
        "zorder": 3,
    },
    "staircase": {
        "label": "Cartesian staircase (HybridFIT3D)",
        "color": "0.25",
        "marker": "D",
        "ls": ":",
        "ms": 5.5,
        "mfc": "0.25",
        "mew": 1.2,
        "zorder": 3,
    },
    "tet": {
        "label": "all-tetrahedral physical region",
        "color": "C3",
        "marker": "P",
        "ls": "-.",
        "ms": 6.8,
        "mfc": "C3",
        "mew": 1.2,
        "zorder": 3,
    },
}

TTS_ANNOT_OFFSETS = {
    ("consistent", 10): ((7, 4), "left", "bottom", False),
    ("consistent", 16): ((8, 3), "left", "bottom", False),
    ("consistent", 20): ((20, 2), "left", "bottom", True),
    ("consistent", 24): ((-8, 6), "right", "bottom", False),
    ("mixed", 10): ((-6, -9), "right", "top", False),
    ("mixed", 16): ((-1, -16), "center", "top", True),
    ("mixed", 20): ((1, 12), "center", "bottom", True),
    ("mixed", 24): ((-8, -6), "right", "top", False),
    ("mixed_smoothed", 10): ((7, -8), "left", "top", False),
    ("mixed_smoothed", 16): ((-8, -4), "right", "center", False),
    ("mixed_smoothed", 20): ((-3, 15), "center", "bottom", True),
    ("mixed_smoothed", 24): ((-15, -12), "right", "top", True),
    ("staircase", 10): ((-7, -9), "right", "top", False),
    ("staircase", 16): ((7, 3), "left", "bottom", False),
    ("staircase", 20): ((-8, -3), "right", "center", False),
    ("staircase", 24): ((-7, 6), "right", "bottom", False),
    ("tet", 10): ((7, 4), "left", "bottom", False),
    ("tet", 16): ((8, -5), "left", "top", False),
    ("tet", 20): ((5, 13), "left", "bottom", True),
    ("tet", 24): ((8, -2), "left", "center", False),
}

COST_ANNOT_OFFSETS = {
    ("consistent", 10): ((-7, 6), "right", "bottom"),
    ("consistent", 16): ((7, 4), "left", "bottom"),
    ("consistent", 20): ((7, 4), "left", "bottom"),
    ("consistent", 24): ((8, -2), "left", "center"),
    ("mixed", 10): ((-7, 1), "right", "center"),
    ("mixed", 16): ((-7, 4), "right", "bottom"),
    ("mixed", 20): ((-8, -8), "right", "top"),
    ("mixed", 24): ((7, -9), "left", "top"),
    ("mixed_smoothed", 10): ((7, -8), "left", "top"),
    ("mixed_smoothed", 16): ((0, -12), "center", "top"),
    ("mixed_smoothed", 20): ((-8, 5), "right", "bottom"),
    ("mixed_smoothed", 24): ((7, 5), "left", "bottom"),
    ("staircase", 10): ((-2, 7), "center", "bottom"),
    ("staircase", 16): ((-2, 7), "center", "bottom"),
    ("staircase", 20): ((-2, 7), "center", "bottom"),
    ("staircase", 24): ((-2, 7), "center", "bottom"),
    ("tet", 10): ((-7, 5), "right", "bottom"),
    ("tet", 16): ((8, -2), "left", "center"),
    ("tet", 20): ((-7, 6), "right", "bottom"),
    ("tet", 24): ((-7, 5), "right", "bottom"),
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
    """Load and merge ``results_mixed_study*.json`` files."""
    if paths is None:
        paths = [p for p in DEFAULT_JSONS if os.path.exists(p)]
    elif isinstance(paths, str):
        paths = [paths]
        if os.path.abspath(paths[0]) == os.path.abspath(DEFAULT_JSON):
            for extra in DEFAULT_JSONS[1:]:
                if os.path.exists(extra):
                    paths.append(extra)
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
            ms=st.get("ms", 6),
            mfc=st.get("mfc", st["color"]),
            mew=st.get("mew", 1.2),
            lw=1.6,
            zorder=st.get("zorder", 3),
        )
        for r in rows:
            n_val = int(r["N"])
            xytext, ha, va, use_arrow = TTS_ANNOT_OFFSETS.get(
                (v, n_val), ((5, 4), "left", "bottom", False)
            )
            arrowprops = (
                {
                    "arrowstyle": "-",
                    "color": st["color"],
                    "lw": 0.6,
                    "alpha": 0.7,
                    "shrinkA": 0,
                    "shrinkB": 3,
                }
                if use_arrow
                else None
            )
            ax.annotate(
                f"N={n_val}",
                (r["far_zone_error"], r["time_to_solution_s"]),
                textcoords="offset points",
                xytext=xytext,
                ha=ha,
                va=va,
                fontsize=7.5,
                color=st["color"],
                arrowprops=arrowprops,
                zorder=6,
            )

    ax.margins(x=0.14, y=0.12)
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
    ax.legend(fontsize=8, loc="lower left")
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
            ms=st.get("ms", 6),
            mfc=st.get("mfc", st["color"]),
            mew=st.get("mew", 1.2),
            lw=1.6,
            zorder=st.get("zorder", 3),
        )
        for r in rows:
            n_val = int(r["N"])
            xytext, ha, va = COST_ANNOT_OFFSETS.get(
                (v, n_val), ((5, 4), "left", "bottom")
            )
            ax.annotate(
                f"N={n_val}",
                (r["n_unknowns"], r["cost_per_unknown_step_s"]),
                textcoords="offset points",
                xytext=xytext,
                ha=ha,
                va=va,
                fontsize=7.5,
                color=st["color"],
                zorder=6,
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
                xytext=(3, -7),
                ha="center",
                va="top",
                fontsize=7.5,
                color="0.35",
                zorder=6,
            )

    ax.margins(x=0.12, y=0.14)
    ax.set_xlabel(
        "Number of unknowns $n_{\\mathrm{unknowns}} = n_{\\mathrm{free}} + n_{\\mathrm{faces}}$ [dimensionless]"
    )
    ax.set_ylabel("Cost per unknown-step [s / (unknown $\\cdot$ step)]")
    ax.set_title(
        "PEC sphere time-domain benchmark (200 steps, 1 core): cost per unknown-step vs unknowns",
        fontsize=10,
    )
    ax.legend(fontsize=8, loc="lower right", bbox_to_anchor=(0.98, 0.17))
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
            "Paths to study JSON files (default: results_mixed_study.json, "
            "results_mixed_study_N20.json, and results_mixed_study_N24.json "
            "merged by key 'table')."
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
