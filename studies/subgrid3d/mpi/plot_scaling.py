# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Plot figures for the 3D subgrid MPI strong-scaling and verification study.

Generates:
* ``subgrid3d_mpi_correctness.png``: bitwise field agreement, Higham energy
  allreduce error vs bound, and weighted work imbalance / halo sizes per case.
* ``subgrid3d_mpi_scaling.png``: strong scaling (steps/s and speedup vs ``P``
  with ideal linear reference and serial `SubgridLTSFIT3D(kernels="csr")`
  baseline) for the large-patch (Part D) and small-patch (Part E) loaded
  cavities under global ``dt`` and LTS, with ``rcb`` and ``blocks`` partitions.

Usage:
    python studies/subgrid3d/mpi/plot_scaling.py --in DIR --out DIR
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _load_study(in_dir):
    for name in ("scaling.json", "scaling_quick.json"):
        path = os.path.join(in_dir, name)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                return json.load(fh), path
    raise FileNotFoundError(
        f"Neither scaling.json nor scaling_quick.json found in {in_dir!r}"
    )


def _select(runs, case, kind, method):
    sub = [
        r
        for r in runs
        if r["case"] == case and r["kind"] == kind and r["method"] == method
    ]
    return sorted(sub, key=lambda r: r["P"])


def plot_correctness(data, out_dir):
    runs = data["runs"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)

    # Panel 1: Bitwise field error (0) and Higham energy relative error vs P
    ax = axes[0]
    markers = {
        ("big", "global"): "o",
        ("big", "lts"): "s",
        ("small", "global"): "^",
        ("small", "lts"): "D",
    }
    colors = {
        ("big", "global"): "#1f77b4",
        ("big", "lts"): "#d62728",
        ("small", "global"): "#2ca02c",
        ("small", "lts"): "#9467bd",
    }
    for (case, kind), mk in markers.items():
        sub = _select(runs, case, kind, "rcb")
        if not sub:
            continue
        ps = [r["P"] for r in sub]
        e_rel = [max(r["energy_rel_diff_20_steps"], 1e-18) for r in sub]
        f_diff = [max(r["max_abs_diff_20_steps"], 1e-20) for r in sub]
        lbl = f"{case} ({kind})"
        ax.semilogy(
            ps,
            e_rel,
            marker=mk,
            color=colors[(case, kind)],
            lw=1.6,
            label=f"|ΔE|/E {lbl}",
        )
        ax.semilogy(
            ps,
            f_diff,
            marker=mk,
            color=colors[(case, kind)],
            lw=1.0,
            ls="--",
            alpha=0.7,
        )
    ax.axhline(2.0**-52, color="k", ls=":", lw=1.0, label="2u (machine eps)")
    ax.set_xlabel("MPI ranks P")
    ax.set_ylabel("Relative difference vs serial")
    ax.set_title("20-step bitwise field (dashed=0) & energy")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7.5, loc="upper left")

    # Panel 2: Weighted work imbalance (rcb vs blocks)
    ax = axes[1]
    for case, ls in [("big", "-"), ("small", "--")]:
        for kind, col in [("global", "#1f77b4"), ("lts", "#d62728")]:
            for method, mk in [("rcb", "o"), ("blocks", "x")]:
                sub = _select(runs, case, kind, method)
                if not sub:
                    continue
                ax.plot(
                    [r["P"] for r in sub],
                    [r["imbalance"] for r in sub],
                    ls=ls,
                    marker=mk,
                    color=col if method == "rcb" else "#ff7f0e",
                    lw=1.5,
                    label=f"{case} {kind} ({method})",
                )
    ax.axhline(1.0, color="k", ls=":", lw=1.0)
    ax.set_xlabel("MPI ranks P")
    ax.set_ylabel("Weighted work imbalance (max / mean)")
    ax.set_title("Work balance: weighted RCB vs whole-block")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7.0, ncol=2)

    # Panel 3: Total halo entries and messages per coarse step (RCB)
    ax = axes[2]
    for (case, kind), col in colors.items():
        sub = _select(runs, case, kind, "rcb")
        if not sub:
            continue
        ps = [r["P"] for r in sub]
        halos = [r["halo_e_sum"] + r["halo_b_sum"] + r["halo_lts_sum"] for r in sub]
        ax.plot(
            ps,
            halos,
            marker=markers[(case, kind)],
            color=col,
            lw=1.6,
            label=f"{case} {kind}",
        )
    ax.set_xlabel("MPI ranks P")
    ax.set_ylabel("Total halo unknowns across ranks")
    ax.set_title("Halo footprint vs MPI ranks P (RCB)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8.0)

    out_path = os.path.join(out_dir, "subgrid3d_mpi_correctness.png")
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def plot_scaling(data, out_dir):
    runs = data["runs"]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2), constrained_layout=True)

    for col_idx, (case, title_tag) in enumerate(
        [
            ("big", "Part D: Large patch (cube + 0.1 m)"),
            ("small", "Part E: Small patch (0.8% vol.)"),
        ]
    ):
        # Top row: Throughput (steps/s) vs P with ideal line from serial baseline
        ax_top = axes[0, col_idx]
        # Bottom row: Speedup vs serial baseline & communication fraction
        ax_bot = axes[1, col_idx]

        p_all = sorted({r["P"] for r in runs if r["case"] == case})
        p_arr = np.asarray(p_all, dtype=float)

        styles = [
            ("global", "rcb", "#1f77b4", "-", "o", "global Δt, RCB"),
            ("global", "blocks", "#1f77b4", "--", "s", "global Δt, blocks"),
            ("lts", "rcb", "#d62728", "-", "^", "LTS, RCB"),
            ("lts", "blocks", "#d62728", "--", "D", "LTS, blocks"),
        ]
        for kind, method, color, ls, mk, lbl in styles:
            sub = _select(runs, case, kind, method)
            if not sub:
                continue
            ps = np.array([r["P"] for r in sub], dtype=float)
            sps = np.array([r["steps_per_s"] for r in sub], dtype=float)
            spdup = np.array([r["speedup_vs_serial"] for r in sub], dtype=float)
            ax_top.loglog(ps, sps, color=color, ls=ls, marker=mk, lw=1.7, label=lbl)
            ax_bot.plot(ps, spdup, color=color, ls=ls, marker=mk, lw=1.7, label=lbl)

            if method == "rcb":
                s_base = sub[0]["serial_steps_per_s"]
                ax_top.loglog(
                    p_arr,
                    s_base * p_arr,
                    color=color,
                    ls=":",
                    lw=1.1,
                    alpha=0.7,
                    label=f"ideal {kind} (from serial)",
                )

        if p_all:
            ax_bot.plot(p_arr, p_arr, color="k", ls=":", lw=1.2, label="ideal linear")
        ax_top.set_xlabel("MPI ranks P")
        ax_top.set_ylabel("Throughput [coarse steps / s]")
        ax_top.set_title(f"{title_tag}: throughput")
        ax_top.grid(True, which="both", alpha=0.3)
        ax_top.legend(fontsize=7.5)

        ax_bot.set_xlabel("MPI ranks P")
        ax_bot.set_ylabel("Speedup vs serial SubgridLTSFIT3D(csr)")
        ax_bot.set_title(f"{title_tag}: strong-scaling speedup")
        ax_bot.grid(True, alpha=0.3)
        ax_bot.legend(fontsize=7.5)

    out_path = os.path.join(out_dir, "subgrid3d_mpi_scaling.png")
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--in", dest="in_dir", required=True, help="Directory with scaling JSON."
    )
    parser.add_argument(
        "--out", dest="out_dir", required=True, help="Output directory for PNGs."
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    data, src_path = _load_study(args.in_dir)
    f1 = plot_correctness(data, args.out_dir)
    f2 = plot_scaling(data, args.out_dir)
    print(f"Loaded {src_path} -> wrote {f1} and {f2}")


if __name__ == "__main__":
    main()
