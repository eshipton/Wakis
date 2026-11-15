"""
Plot topological mesh-repair metrics from `results/repair.json`.

    python plot_repair.py          # writes studies/unstructured-cadmesh/figures/*.png
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results", "repair.json")
FIG = os.path.join(HERE, "figures")

plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3})


def _load_records():
    with open(RES) as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data.get("all_runs") or data.get("cases", [])
    return data


def _label(rec):
    opt = rec.get("optimize", "none")
    return f"{rec['case']}\n({opt})" if opt != "none" else f"{rec['case']}\n(Delaunay)"


def plot_negative_faces(records):
    """Bar chart of negative dual-length faces (fraction of faces) and topological flip counts."""
    labels = [_label(r) for r in records]
    x = np.arange(len(records))
    w = 0.20

    neg_before = [r["negative_faces_before"] / r["n_faces_before"] for r in records]
    neg_flip = [r["negative_faces_flip_only"] / r["n_faces_flip_only"] for r in records]
    neg_false = [
        r["negative_faces_after_repair_false"] / r["n_faces_after_repair_false"]
        for r in records
    ]
    neg_true = [
        r["negative_faces_after_repair_true"] / r["n_faces_after_repair_true"]
        for r in records
    ]

    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.0))
    ax[0].bar(
        x - 1.5 * w,
        neg_before,
        width=w,
        color="#4C72B0",
        label="Before improve_dt",
    )
    ax[0].bar(
        x - 0.5 * w,
        neg_flip,
        width=w,
        color="#8172B3",
        label="flip_to_delaunay only",
    )
    ax[0].bar(
        x + 0.5 * w,
        neg_false,
        width=w,
        color="#DD8452",
        label="improve_dt(repair=False)",
    )
    ax[0].bar(
        x + 1.5 * w,
        neg_true,
        width=w,
        color="#55A868",
        label="improve_dt(repair=True)",
    )
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(labels, fontsize=8)
    ax[0].set_xlabel("CAD geometry (cpw = 10) [-]")
    ax[0].set_ylabel("Negative dual-length faces [fraction of faces]")
    ax[0].set_title("(a) Non-Delaunay faces before and after repair", fontsize=9)
    ax[0].legend(fontsize=7.5)

    flip_types = ("2-3", "3-2", "4-4", "2-2")
    colors = ("#4C72B0", "#55A868", "#C44E52", "#8172B3")
    bottom = np.zeros(len(records))
    for ft, col in zip(flip_types, colors):
        counts = np.array(
            [r.get("repair_report", {}).get("flips", {}).get(ft, 0) for r in records],
            dtype=float,
        )
        ax[1].bar(x, counts, width=0.5, bottom=bottom, color=col, label=f"{ft} flip")
        bottom += counts

    ax[1].set_xticks(x)
    ax[1].set_xticklabels(labels, fontsize=8)
    ax[1].set_xlabel("CAD geometry (cpw = 10) [-]")
    ax[1].set_ylabel("Executed flips [count]")
    ax[1].set_title(
        "(b) Topological flips by type in improve_dt(repair=True)", fontsize=9
    )
    ax[1].legend(fontsize=7.5)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "repair_negative_faces.png"), dpi=170)
    plt.close(fig)


def plot_dt_and_frequency(records):
    """Bar chart of dt gain (repair=True / repair=False) and lowest eigenfrequency error [-]."""
    labels = [_label(r) for r in records]
    x = np.arange(len(records))
    w = 0.25

    dt_gain = [r["dt_gain_vs_repair_false"] for r in records]
    limit_rr = [r.get("limit_radius_ratio") for r in records]
    limit_el = [r.get("limit_element") for r in records]
    limit_dih = [r.get("limit_min_dihedral") for r in records]

    err_before = [abs(r["lowest_freq_err_before"]) for r in records]
    err_false = [abs(r["lowest_freq_err_repair_false"]) for r in records]
    err_true = [abs(r["lowest_freq_err_repair_true"]) for r in records]

    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.0))
    bars = ax[0].bar(
        x,
        dt_gain,
        width=0.45,
        color="#55A868",
        label="dt gain (repair=True / repair=False)",
    )
    ax[0].axhline(1.0, color="#333333", linestyle="--", linewidth=0.8)
    for b, rr, el, dih in zip(bars, limit_rr, limit_el, limit_dih):
        if rr is not None:
            note = f"rr={rr:.2f}"
            if el is not None and dih is not None:
                note = f"T{el}: rr={rr:.2f}"
            ax[0].annotate(
                note,
                (b.get_x() + b.get_width() / 2.0, b.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
            )
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(labels, fontsize=8)
    ax[0].set_xlabel("CAD geometry (cpw = 10) [-]")
    ax[0].set_ylabel("Time-step ratio [dt gain (repair=True / repair=False)]")
    ax[0].set_title("(a) Time-step ratio vs node smoothing alone", fontsize=9)
    ax[0].legend(fontsize=7.5)

    ax[1].bar(x - w, err_before, width=w, color="#4C72B0", label="Before improve_dt")
    ax[1].bar(
        x,
        err_false,
        width=w,
        color="#DD8452",
        label="improve_dt(repair=False)",
    )
    ax[1].bar(
        x + w,
        err_true,
        width=w,
        color="#55A868",
        label="improve_dt(repair=True)",
    )
    ax[1].set_yscale("log")
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(labels, fontsize=8)
    ax[1].set_xlabel("CAD geometry (cpw = 10) [-]")
    ax[1].set_ylabel(r"Lowest-mode relative error $|f_0 / f_{\mathrm{ref}} - 1|$ [-]")
    ax[1].set_title("(b) Lowest eigenfrequency error vs reference", fontsize=9)
    ax[1].legend(fontsize=7.5)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "repair_dt_accuracy.png"), dpi=170)
    plt.close(fig)


def main():
    os.makedirs(FIG, exist_ok=True)
    records = _load_records()
    plot_negative_faces(records)
    plot_dt_and_frequency(records)


if __name__ == "__main__":
    main()
