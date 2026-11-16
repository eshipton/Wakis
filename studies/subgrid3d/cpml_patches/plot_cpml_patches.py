# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Plotting script for W4-3a Part 3 (`studies/subgrid3d/cpml_patches`):
reads `cpml_patches.json` (or `cpml_patches_quick.json`) from `--in DIR` and
writes two figures into `--out DIR` (default `docs/img`):
  - `subgrid3d_cpml_patches_reflection.png`: time-gated reflection error vs
    layer thickness ``n_pml`` across the four configurations (`nopatch`,
    `patch_near`, `patch_through_c1_alpha0`, `patch_through_alpha1GHz`) and
    honest panel showing low-frequency reflection degradation at
    ``alpha / (2 pi) = 1 GHz``;
  - `subgrid3d_cpml_patches_spectrum.png`: semi-discrete generator
    ``max Re lambda / sigma_max`` under `"fine"` vs `"coarse_cell"` (C1) and
    unstable eigenvector energy localisation by block, interface, and depth.
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _load_results(in_dir):
    for name in ("cpml_patches.json", "cpml_patches_quick.json"):
        p = os.path.join(in_dir, name)
        if os.path.isfile(p):
            with open(p) as f:
                return json.load(f)
    raise FileNotFoundError(f"no cpml_patches*.json found in {in_dir}")


def plot_reflection(data, out_dir):
    rows = data["reflection"]
    re_var = data["spectrum"]["reflection_variants_re_over_smax"]
    variants = [
        (
            "nopatch",
            f"nopatch (uniform SubgridLTSFIT3D, stable Re$\\lambda/\\sigma_{{\\max}}$={re_var['nopatch']:.2e})",
            "#1f77b4",
            None,
        ),
        (
            "patch_near",
            f"patch_near (1 cell outside PML, default-allowed, stable {re_var['patch_near']:.2e})",
            "#2ca02c",
            None,
        ),
        (
            "patch_through_c1_alpha0",
            f"patch_through_c1_alpha0 (C1 $\\alpha=0$, SPECTRALLY UNSTABLE {re_var['patch_through_c1_alpha0']:.2e}, strict xfail)",
            "#d62728",
            "///",
        ),
        (
            "patch_through_alpha1GHz",
            f"patch_through_alpha1GHz (C1 $\\alpha/2\\pi=1$ GHz, stable {re_var['patch_through_alpha1GHz']:.2e})",
            "#9467bd",
            None,
        ),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 4.8), constrained_layout=True)

    # Panel (a): Reflection table as a grouped bar chart across n_pml
    npml_vals = sorted({r["n_pml"] for r in rows})
    x = np.arange(len(npml_vals), dtype=float)
    width = 0.19
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(variants))

    unstable_bar_coords = []
    for (key, label, color, hatch), off in zip(variants, offsets):
        sub = {r["n_pml"]: r["err"] for r in rows if r["variant"] == key}
        ys = [sub.get(n, np.nan) for n in npml_vals]
        bars = ax1.bar(
            x + off,
            ys,
            width=width,
            color=color,
            edgecolor="#111111" if hatch else "none",
            hatch=hatch,
            alpha=0.88 if hatch else 0.92,
            label=label,
        )
        if key == "patch_through_c1_alpha0":
            for b_item in bars:
                unstable_bar_coords.append(
                    (
                        b_item.get_x() + 0.5 * b_item.get_width(),
                        b_item.get_height(),
                    )
                )

    if unstable_bar_coords:
        bx, by = unstable_bar_coords[min(1, len(unstable_bar_coords) - 1)]
        ax1.annotate(
            "SPECTRALLY UNSTABLE ROW\n"
            f"(patch_through_c1_alpha0: Re$\\lambda/\\sigma_{{\\max}}={re_var['patch_through_c1_alpha0']:.2e} > 10^{{-6}}$;\n"
            "refused by default allow_patches=False)",
            xy=(bx, by),
            xytext=(bx + 0.12, max(by * 5.5, 2.5e-2)),
            fontsize=7.3,
            fontweight="bold",
            color="#8b0000",
            bbox={
                "boxstyle": "round,pad=0.25",
                "fc": "#fff0f0",
                "ec": "#d62728",
                "lw": 1.2,
            },
            arrowprops={
                "arrowstyle": "->",
                "color": "#d62728",
                "lw": 1.4,
            },
        )

    ax1.set_yscale("log")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"$n_{{\\mathrm{{pml}}}} = {n}$" for n in npml_vals])
    ax1.set_xlabel("CPML layer thickness $n_{\\mathrm{pml}}$ [coarse cells]")
    ax1.set_ylabel(
        "Time-gated relative reflection error $\\max|e_{\\mathrm{open}} - e_{\\mathrm{ref}}| / \\max|e_{\\mathrm{ref}}|$ [1]"
    )
    ax1.set_title(
        "(a) CPML reflection table vs layer thickness (hatched = spectrally unstable)",
        fontsize=10.2,
    )
    ax1.grid(True, axis="y", which="both", ls=":", alpha=0.6)
    ax1.legend(fontsize=7.1, loc="lower left")

    # Panel (b): Honest panel — probe waveform difference |e_open - e_ref| at largest n_pml
    max_npml = max(r["n_pml"] for r in rows)
    line_styles = {
        "nopatch": ("-", "#1f77b4"),
        "patch_near": ("--", "#2ca02c"),
        "patch_through_c1_alpha0": ("-.", "#d62728"),
        "patch_through_alpha1GHz": (":", "#9467bd"),
    }
    short_labels = {
        "nopatch": "nopatch (stable)",
        "patch_near": "patch_near (stable)",
        "patch_through_c1_alpha0": "patch_through_c1_alpha0 (UNSTABLE at late time)",
        "patch_through_alpha1GHz": "patch_through_alpha1GHz ($\\alpha/2\\pi=1$ GHz, stable)",
    }
    for key, _, _, _ in variants:
        matches = [r for r in rows if r["variant"] == key and r["n_pml"] == max_npml]
        if not matches:
            continue
        r0 = matches[0]
        t_ns = 1e9 * np.asarray(r0["t"], float)
        sig_open = np.asarray(r0["sig_open"], float)
        sig_ref = np.asarray(r0["sig_ref"], float)
        ref_pk = np.max(np.abs(sig_ref)) + 1e-30
        diff = np.abs(sig_open - sig_ref) / ref_pk
        ls, col = line_styles[key]
        ax2.semilogy(
            t_ns,
            np.maximum(diff, 1e-8),
            ls=ls,
            color=col,
            lw=1.7,
            label=f"{short_labels[key]} ($n_{{\\mathrm{{pml}}}}={max_npml}$)",
        )

    ax2.set_xlabel("Time $t$ [ns]")
    ax2.set_ylabel(
        "Normalized reflection "
        "$|e_{\\mathrm{open}}(t) - e_{\\mathrm{ref}}(t)| / \\max|e_{\\mathrm{ref}}|$ [1]"
    )
    ax2.set_title(
        f"(b) Honest panel ($n_{{\\mathrm{{pml}}}}={max_npml}$): $\\alpha/2\\pi=1$ GHz reflects; $\\alpha=0$ unstable late",
        fontsize=10.2,
    )
    ax2.grid(True, which="both", ls=":", alpha=0.6)
    ax2.legend(fontsize=7.6, loc="lower right")

    out_path = os.path.join(out_dir, "subgrid3d_cpml_patches_reflection.png")
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def plot_spectrum(data, out_dir):
    spec = data["spectrum"]
    re_map = spec["re_over_smax"]
    diag = spec["diagnosis_fine"]
    t1_rows = spec.get("t1_geometries", [])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 4.8), constrained_layout=True)

    if t1_rows:
        geom_names = [r["geometry"] for r in t1_rows] + ["8x5x5_into", "8x5x5_thru"]
        fine_vals = [max(abs(r["re_fine_10m"]), 1e-16) for r in t1_rows] + [
            max(abs(re_map["patch_into_fine"]), 1e-16),
            max(abs(re_map["patch_through_fine"]), 1e-16),
        ]
        c1_vals = [max(abs(r["re_c1_10m"]), 1e-16) for r in t1_rows] + [
            max(abs(re_map["patch_into_coarse_cell"]), 1e-16),
            max(abs(re_map["patch_through_coarse_cell"]), 1e-16),
        ]
        c1_stable = [bool(r["stable_c1_10m"]) for r in t1_rows] + [
            re_map["patch_into_coarse_cell"] <= 1e-6,
            re_map["patch_through_coarse_cell"] <= 1e-6,
        ]
    else:
        geom_names = ["nopatch", "8x5x5_into", "8x5x5_thru"]
        fine_vals = [
            max(abs(re_map["nopatch"]), 1e-16),
            max(abs(re_map["patch_into_fine"]), 1e-16),
            max(abs(re_map["patch_through_fine"]), 1e-16),
        ]
        c1_vals = [
            max(abs(re_map["nopatch"]), 1e-16),
            max(abs(re_map["patch_into_coarse_cell"]), 1e-16),
            max(abs(re_map["patch_through_coarse_cell"]), 1e-16),
        ]
        c1_stable = [v <= 1e-6 for v in c1_vals]

    x = np.arange(len(geom_names), dtype=float)
    w = 0.38
    ax1.bar(
        x - 0.5 * w,
        fine_vals,
        width=w,
        color="#d62728",
        alpha=0.85,
        label="Fine sampling (sigma_sampling='fine' — UNSTABLE $> 5\\times 10^{-3}$)",
    )
    # Plot C1 stable vs C1 unstable separately so legend marks STABLE vs UNSTABLE clearly
    idx_st = [i for i, st in enumerate(c1_stable) if st]
    idx_un = [i for i, st in enumerate(c1_stable) if not st]
    if idx_st:
        ax1.bar(
            x[idx_st] + 0.5 * w,
            [c1_vals[i] for i in idx_st],
            width=w,
            color="#2ca02c",
            alpha=0.92,
            label="Candidate C1 (coarse_cell) — STABLE ($\\leq 10^{-6}\\sigma_{\\max}$)",
        )
    if idx_un:
        ax1.bar(
            x[idx_un] + 0.5 * w,
            [c1_vals[i] for i in idx_un],
            width=w,
            color="#ff7f0e",
            edgecolor="#111111",
            hatch="///",
            alpha=0.92,
            label="Candidate C1 (coarse_cell) — UNSTABLE ($> 10^{-6}\\sigma_{\\max}$, strict xfail)",
        )

    ax1.axhline(
        1e-6,
        color="#111111",
        ls="--",
        lw=1.3,
        label="T1 stability threshold ($10^{-6}\\sigma_{\\max}$)",
    )
    ax1.set_yscale("log")
    ax1.set_ylim(1e-17, 1e-1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(geom_names, rotation=32, ha="right", fontsize=7.6)
    ax1.set_ylabel(
        "Normalized growth rate $\\max\\,\\mathrm{Re}\\,\\lambda(G) / \\sigma_{\\max}$ [1]"
    )
    ax1.set_title(
        "(a) Generator spectrum per geometry ($\\alpha/2\\pi = 10$ MHz; refused by default)"
    )
    ax1.grid(True, axis="y", which="both", ls=":", alpha=0.6)
    ax1.legend(fontsize=6.9, loc="lower left")

    # Panel (b): Unstable eigenvector localisation for fine sigma_sampling
    loc_labels = [
        "Coarse block 0\n(EM energy)",
        "Patch block 1\n(EM energy)",
        "Interface faces\n(share of $E_b$)",
        "Interface edges\n(share of $E_e$)",
        "Depth Q1\n($u \\in (0, 0.25]$)",
        "Depth Q2\n($u \\in (0.25, 0.5]$)",
    ]
    b_shares = diag["block_energy_share"]
    b0 = b_shares.get("0", b_shares.get(0, 0.0))
    b1 = b_shares.get("1", b_shares.get(1, 0.0))
    q = diag["depth_quartile_share"]
    loc_vals = [
        100.0 * b0,
        100.0 * b1,
        100.0 * diag["interface_face_share_of_b"],
        100.0 * diag["interface_edge_share_of_e"],
        100.0 * q[0],
        100.0 * q[1],
    ]
    x2 = np.arange(len(loc_labels))
    bars2 = ax2.bar(
        x2,
        loc_vals,
        color=["#1f77b4", "#ff7f0e", "#d62728", "#d62728", "#9467bd", "#9467bd"],
        width=0.58,
    )
    for b_item, val in zip(bars2, loc_vals):
        ax2.text(
            b_item.get_x() + 0.5 * b_item.get_width(),
            val + 1.8,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=7.8,
            fontweight="bold",
        )
    ax2.set_xticks(x2)
    ax2.set_xticklabels(loc_labels, fontsize=8.0)
    ax2.set_ylabel("Modal energy share [%]")
    ax2.set_ylim(0, 108)
    ax2.set_title("(b) Unstable eigenvector localisation (sigma_sampling='fine')")
    ax2.grid(True, axis="y", ls=":", alpha=0.6)

    out_path = os.path.join(out_dir, "subgrid3d_cpml_patches_spectrum.png")
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_dir", required=True, help="Input directory")
    ap.add_argument(
        "--out",
        dest="out_dir",
        default=os.path.join("docs", "img"),
        help="Output directory for PNG figures",
    )
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    data = _load_results(a.in_dir)
    p1 = plot_reflection(data, a.out_dir)
    p2 = plot_spectrum(data, a.out_dir)
    print(f"wrote {p1} and {p2}")


if __name__ == "__main__":
    main()
