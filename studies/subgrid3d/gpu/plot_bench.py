# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Plot benchmark figures for fused 3D subgridding GPU kernels
(`studies/subgrid3d/gpu/bench_fused.py`).

Generates two figures in `--out DIR`:

1. `subgrid3d_gpu_ms_per_step.png`:
   * Left panel: uniform PEC loaded-cavity grid (`H = 0.1 ... 0.0125 m`),
     ms/step vs unknowns for `SolverFIT3D` (CSR and fused), `SubgridLTSFIT3D`
     (`csr` and `block` on CuPy), and `FusedSubgridStepper` (direct and CUDA
     graph).
   * Right panel: 2:1 and 3:1 subgrid loaded-cavity meshes (global `dt` and
     LTS), ms/step vs unknowns for `csr`, `block`, `fused (direct)` and
     `fused (graph)`.
2. `subgrid3d_gpu_roofline.png`:
   * Left panel: measured ms/step vs compulsory-traffic roofline prediction
     `B_step / BW_copy` across uniform, global-`dt` subgrid and LTS subgrid
     runs.
   * Right panel: ns per unknown-step and time to physical duration `T = 10 ns`
     across all subgrid GPU arms.

Usage:
    python studies/subgrid3d/gpu/plot_bench.py --in DIR --out DIR
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _load_bench(in_dir):
    for name in ("bench.json", "bench_quick.json"):
        p = os.path.join(in_dir, name)
        if os.path.exists(p):
            with open(p) as f:
                return json.load(f), p
    raise FileNotFoundError(
        f"neither bench.json nor bench_quick.json found in {in_dir}"
    )


def plot_ms_per_step(data, out_dir):
    rows = data["rows"]
    gpu_name = data["meta"].get("gpu_name", "GPU")

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12.0, 4.8))

    # Left panel: uniform grid arms
    uni = sorted(
        [r for r in rows if r["kind"] == "uniform"], key=lambda r: r["unknowns"]
    )
    if uni:
        u_dof = [r["unknowns"] for r in uni]
        arms_uni = [
            ("fit3d_csr_ms", "SolverFIT3D (CSR)", "s--", "#7f7f7f"),
            ("fit3d_fused_ms", "SolverFIT3D (fused)", "o-", "#1f77b4"),
            ("subgrid_block_ms", "Subgrid block (CuPy)", "^--", "#d62728"),
            ("subgrid_csr_ms", "Subgrid CSR (CuPy)", "v--", "#ff7f0e"),
            ("subgrid_fused_direct_ms", "Subgrid fused (direct)", "d-", "#2ca02c"),
            ("subgrid_fused_graph_ms", "Subgrid fused (CUDA graph)", "*-", "#17becf"),
        ]
        for key, label, style, color in arms_uni:
            if all(key in r for r in uni):
                ax0.loglog(
                    u_dof,
                    [r[key] for r in uni],
                    style,
                    color=color,
                    label=label,
                    linewidth=1.6,
                    markersize=5,
                )
    ax0.set_xlabel("Unknowns ($n_{\\mathrm{free}} + n_{\\mathrm{faces}}$)")
    ax0.set_ylabel("Wall time per step [ms/step]")
    ax0.set_title(f"Uniform grid ({gpu_name})")
    ax0.grid(True, which="both", alpha=0.3)
    ax0.legend(fontsize=8)

    # Right panel: 2:1 and 3:1 subgrid loaded-cavity arms
    sg = sorted(
        [r for r in rows if r["kind"] != "uniform"], key=lambda r: r["unknowns"]
    )
    if sg:
        s_dof = [r["unknowns"] for r in sg]
        arms_sg = [
            ("subgrid_block_ms", "Subgrid block (CuPy)", "^--", "#d62728"),
            ("subgrid_csr_ms", "Subgrid CSR (CuPy)", "v--", "#ff7f0e"),
            ("subgrid_fused_direct_ms", "Subgrid fused (direct)", "d-", "#2ca02c"),
            ("subgrid_fused_graph_ms", "Subgrid fused (CUDA graph)", "*-", "#17becf"),
        ]
        for key, label, style, color in arms_sg:
            ax1.loglog(
                s_dof,
                [r[key] for r in sg],
                style,
                color=color,
                label=label,
                linewidth=1.6,
                markersize=5,
            )
    ax1.set_xlabel("Unknowns ($n_{\\mathrm{free}} + n_{\\mathrm{faces}}$)")
    ax1.set_ylabel("Wall time per coarse step [ms/step]")
    ax1.set_title(f"Subgrid 2:1 and 3:1 loaded cavity ({gpu_name})")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(fontsize=8)

    fig.tight_layout()
    out_path = os.path.join(out_dir, "subgrid3d_gpu_ms_per_step.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_roofline(data, out_dir):
    rows = data["rows"]
    bw = data["meta"].get("copy_bw_gb_s", 0.0)
    gpu_name = data["meta"].get("gpu_name", "GPU")

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12.0, 4.8))

    # Left panel: measured vs roofline ms/step
    for kind, marker, color, label in [
        ("uniform", "o", "#1f77b4", "Uniform (global $\\Delta t$, 2 kernels)"),
        ("global", "s", "#2ca02c", "Subgrid global $\\Delta t$ (2 kernels)"),
        ("lts", "^", "#d62728", "Subgrid LTS ($2P$ kernels)"),
    ]:
        sub = sorted(
            [r for r in rows if r["kind"] == kind], key=lambda r: r["unknowns"]
        )
        if not sub:
            continue
        dof = [r["unknowns"] for r in sub]
        meas = [r["subgrid_fused_graph_ms"] for r in sub]
        roof = [r["roofline_ms"] for r in sub]
        ax0.loglog(
            dof,
            meas,
            f"{marker}-",
            color=color,
            label=f"{label} measured",
            linewidth=1.6,
        )
        ax0.loglog(
            dof,
            roof,
            f"{marker}--",
            color=color,
            alpha=0.65,
            label=f"{label} roofline",
            linewidth=1.2,
        )

    ax0.set_xlabel("Unknowns ($n_{\\mathrm{free}} + n_{\\mathrm{faces}}$)")
    ax0.set_ylabel("Time per coarse step [ms/step]")
    ax0.set_title(f"Measured vs roofline ({gpu_name}, copy BW = {bw:.1f} GB/s)")
    ax0.grid(True, which="both", alpha=0.3)
    ax0.legend(fontsize=7.5)

    # Right panel: ns per unknown-step across all runs
    all_sorted = sorted(rows, key=lambda r: (r["kind"], r["unknowns"]))
    x_idx = np.arange(len(all_sorted))
    labels = [
        f"{r['kind'][:3]}\nH={r['h_coarse']:.3g},r={r['ratio']}" for r in all_sorted
    ]
    ax1.semilogy(
        x_idx,
        [r["subgrid_block_ns_per_dof"] for r in all_sorted],
        "^--",
        color="#d62728",
        label="Block (CuPy)",
    )
    ax1.semilogy(
        x_idx,
        [r["subgrid_csr_ns_per_dof"] for r in all_sorted],
        "v--",
        color="#ff7f0e",
        label="CSR (CuPy)",
    )
    ax1.semilogy(
        x_idx,
        [r["subgrid_fused_direct_ns_per_dof"] for r in all_sorted],
        "d-",
        color="#2ca02c",
        label="Fused (direct)",
    )
    ax1.semilogy(
        x_idx,
        [r["subgrid_fused_graph_ns_per_dof"] for r in all_sorted],
        "*-",
        color="#17becf",
        label="Fused (CUDA graph)",
    )
    ax1.set_xticks(x_idx)
    ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=7.5)
    ax1.set_ylabel("Time per unknown-step [ns / (unknown $\\cdot$ step)]")
    ax1.set_title("Throughput per unknown-step")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(fontsize=8)

    fig.tight_layout()
    out_path = os.path.join(out_dir, "subgrid3d_gpu_roofline.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Plot 3D subgrid GPU benchmark figures.")
    ap.add_argument(
        "--in", dest="in_dir", required=True, help="Directory with bench[_quick].json."
    )
    ap.add_argument(
        "--out", dest="out_dir", required=True, help="Output directory for PNGs."
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    data, src_path = _load_bench(args.in_dir)
    p1 = plot_ms_per_step(data, args.out_dir)
    p2 = plot_roofline(data, args.out_dir)
    print(f"loaded {src_path} -> wrote {p1}, {p2}")


if __name__ == "__main__":
    main()
