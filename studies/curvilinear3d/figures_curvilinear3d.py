# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Generate convergence, equal-unknowns, and equal-wall-time figures from the JSON
output of `studies/curvilinear3d/study_cavities.py`.

Uses the non-interactive matplotlib ``Agg`` backend and labels all axes with
physical units.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _characteristic_h(rec):
    """Characteristic grid spacing ``h = L_char / N`` [m] for a study record."""
    n = float(rec["n_per_axis"])
    geom = rec["geometry"]
    if geom == "perturbed_box":
        return 1.0 / n
    return 2.0 / n


def plot_convergence(study_data, out_path):
    """Plot relative eigenfrequency error vs characteristic spacing ``h`` [m],
    active unknown edge count ``n_unknowns`` [1], and wall-clock time ``wall_s``
    [s] across all geometries and modes."""
    records = study_data["records"]
    geom_modes = []
    for r in records:
        gm = (r["geometry"], r["mode"])
        if gm not in geom_modes:
            geom_modes.append(gm)

    n_rows = max(len(geom_modes), 1)
    fig, axes = plt.subplots(
        n_rows, 3, figsize=(14.0, 3.4 * n_rows), squeeze=False, constrained_layout=True
    )

    styles = {
        "CurvilinearFIT3D": {"marker": "o", "linestyle": "-"},
        "SolverFIT3D staircase": {"marker": "s", "linestyle": "--"},
        "SolverFIT3D conformal": {"marker": "^", "linestyle": "-."},
    }

    for row_idx, (geom, mode) in enumerate(geom_modes):
        ax_h, ax_u, ax_t = axes[row_idx]
        sub = [r for r in records if r["geometry"] == geom and r["mode"] == mode]
        methods = []
        for r in sub:
            if r["method"] not in methods:
                methods.append(r["method"])

        for method in methods:
            m_recs = sorted(
                [r for r in sub if r["method"] == method],
                key=lambda r: r["n_per_axis"],
            )
            hs = np.array([_characteristic_h(r) for r in m_recs], dtype=float)
            unks = np.array([r["n_unknowns"] for r in m_recs], dtype=float)
            walls = np.array([r["wall_s"] for r in m_recs], dtype=float)
            errs = np.array([r["rel_error"] for r in m_recs], dtype=float)
            order = m_recs[0].get("order")
            label = (
                f"{method} (p={order:.2f})"
                if order is not None and np.isfinite(order)
                else method
            )
            st = styles.get(method, {"marker": "o", "linestyle": "-"})
            ax_h.loglog(hs, errs, label=label, **st)
            ax_u.loglog(unks, errs, label=label, **st)
            ax_t.loglog(walls, errs, label=label, **st)

        title_prefix = f"{geom} ({mode})"
        ax_h.set_title(f"{title_prefix}: error vs spacing")
        ax_h.set_xlabel("Characteristic spacing h [m]")
        ax_h.set_ylabel("Relative error |omega_h - omega_ana| / omega_ana [1]")
        ax_h.grid(True, which="both", alpha=0.3)
        ax_h.legend(fontsize=8)

        ax_u.set_title(f"{title_prefix}: equal-unknowns")
        ax_u.set_xlabel("Active unknown edges n_unknowns [1]")
        ax_u.set_ylabel("Relative error |omega_h - omega_ana| / omega_ana [1]")
        ax_u.grid(True, which="both", alpha=0.3)
        ax_u.legend(fontsize=8)

        ax_t.set_title(f"{title_prefix}: equal wall time")
        ax_t.set_xlabel("Wall-clock time [s]")
        ax_t.set_ylabel("Relative error |omega_h - omega_ana| / omega_ana [1]")
        ax_t.grid(True, which="both", alpha=0.3)
        ax_t.legend(fontsize=8)

    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Plot 3D curvilinear cavity study figures (matplotlib Agg)."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to JSON output produced by study_cavities.py.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="figures_curvilinear3d.png",
        help="Output PNG figure path.",
    )
    args = parser.parse_args(argv)

    in_path = Path(args.input)
    with in_path.open("r", encoding="utf-8") as fh:
        study_data = json.load(fh)

    plot_convergence(study_data, Path(args.out))


if __name__ == "__main__":
    main()
