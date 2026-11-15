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
    """Bar chart of negative dual-length faces and topological flip counts."""
    labels = [_label(r) for r in records]
    x = np.arange(len(records))
    w = 0.25

    neg_before = [r["negative_faces_before"] for r in records]
    neg_false = [r["negative_faces_after_repair_false"] for r in records]
    neg_true = [r["negative_faces_after_repair_true"] for r in records]

    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.0))
    ax[0].bar(x - w, neg_before, width=w, color="#4C72B0", label="Before improve_dt")
    ax[0].bar(
        x,
        neg_false,
        width=w,
        color="#DD8452",
        label="improve_dt(repair=False)",
    )
    ax[0].bar(
        x + w,
        neg_true,
        width=w,
        color="#55A868",
        label="improve_dt(repair=True)",
    )
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(labels, fontsize=8)
    ax[0].set_xlabel("CAD geometry (cpw = 10) [-]")
    ax[0].set_ylabel("Negative dual-length faces [count]")
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
    """Bar chart of stable time step [ps] and lowest eigenfrequency error [-]."""
    labels = [_label(r) for r in records]
    x = np.arange(len(records))
    w = 0.25

    dt_before_ps = [r["dt_max_before"] * 1e12 for r in records]
    dt_false_ps = [r["dt_max_repair_false"] * 1e12 for r in records]
    dt_true_ps = [r["dt_max_repair_true"] * 1e12 for r in records]

    err_before = [abs(r["lowest_freq_err_before"]) for r in records]
    err_false = [abs(r["lowest_freq_err_repair_false"]) for r in records]
    err_true = [abs(r["lowest_freq_err_repair_true"]) for r in records]

    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.0))
    ax[0].bar(x - w, dt_before_ps, width=w, color="#4C72B0", label="Before improve_dt")
    ax[0].bar(
        x,
        dt_false_ps,
        width=w,
        color="#DD8452",
        label="improve_dt(repair=False)",
    )
    ax[0].bar(
        x + w,
        dt_true_ps,
        width=w,
        color="#55A868",
        label="improve_dt(repair=True)",
    )
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(labels, fontsize=8)
    ax[0].set_xlabel("CAD geometry (cpw = 10) [-]")
    ax[0].set_ylabel(r"Maximum stable step $\Delta t_{\max}$ [ps]")
    ax[0].set_title("(a) Stable leapfrog time step", fontsize=9)
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
