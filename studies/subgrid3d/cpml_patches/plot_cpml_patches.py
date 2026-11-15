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
    variants = [
        ("nopatch", "No patch", "o-", "C0"),
        ("patch_near", "Patch 1 cell away", "s--", "C1"),
        ("patch_through_c1_alpha0", "Patch through (C1, $\\alpha=0$)", "^-", "C2"),
        (
            "patch_through_alpha1GHz",
            "Patch through ($\\alpha/2\\pi=1$ GHz)",
            "v:",
            "C3",
        ),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))

    for key, label, style, color in variants:
        sub = sorted([r for r in rows if r["variant"] == key], key=lambda r: r["n_pml"])
        if not sub:
            continue
        xs = [r["n_pml"] for r in sub]
        ys = [r["err"] for r in sub]
        ax1.semilogy(xs, ys, style, color=color, label=label, lw=1.8, ms=6)

    ax1.set_xlabel("CPML layer thickness $n_{\\mathrm{pml}}$ [coarse cells]")
    ax1.set_ylabel("Time-gated relative reflection error [-]")
    ax1.set_title("CPML Reflection vs. Layer Thickness")
    ax1.grid(True, which="both", ls=":", alpha=0.6)
    ax1.legend(fontsize=8.5)

    # Panel 2 (honest panel): probe waveform difference |e_open - e_ref| at the
    # largest n_pml, illustrating low-frequency reflection when alpha/2pi = 1 GHz
    max_npml = max(r["n_pml"] for r in rows)
    for key, label, _, color in variants:
        matches = [r for r in rows if r["variant"] == key and r["n_pml"] == max_npml]
        if not matches:
            continue
        r0 = matches[0]
        t_ns = 1e9 * np.asarray(r0["t"], float)
        sig_open = np.asarray(r0["sig_open"], float)
        sig_ref = np.asarray(r0["sig_ref"], float)
        ref_pk = np.max(np.abs(sig_ref)) + 1e-30
        diff = np.abs(sig_open - sig_ref) / ref_pk
        ax2.semilogy(
            t_ns,
            np.maximum(diff, 1e-8),
            color=color,
            lw=1.5,
            label=f"{label} ($n_{{\\mathrm{{pml}}}}={max_npml}$)",
        )

    ax2.set_xlabel("Time $t$ [ns]")
    ax2.set_ylabel(
        "Normalized reflection "
        "$|e_{\\mathrm{open}} - e_{\\mathrm{ref}}| / \\max|e_{\\mathrm{ref}}|$ [-]"
    )
    ax2.set_title("Honest Panel: CFS $\\alpha/2\\pi=1$ GHz Reflects Low Frequencies")
    ax2.grid(True, which="both", ls=":", alpha=0.6)
    ax2.legend(fontsize=8)

    fig.tight_layout()
    out_path = os.path.join(out_dir, "subgrid3d_cpml_patches_reflection.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_spectrum(data, out_dir):
    spec = data["spectrum"]
    re_map = spec["re_over_smax"]
    diag = spec["diagnosis_fine"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))

    labels = [
        "No patch",
        "Patch into\n(fine $\\sigma$)",
        "Patch into\n(C1 coarse cell)",
        "Patch through\n(fine $\\sigma$)",
        "Patch through\n(C1 coarse cell)",
    ]
    keys = [
        "nopatch",
        "patch_into_fine",
        "patch_into_coarse_cell",
        "patch_through_fine",
        "patch_through_coarse_cell",
    ]
    vals = [max(abs(re_map[k]), 1e-16) for k in keys]
    colors = ["C0", "C3", "C2", "C3", "C1"]

    x = np.arange(len(labels))
    ax1.bar(x, vals, color=colors, width=0.58)
    ax1.axhline(
        1e-6,
        color="k",
        ls="--",
        lw=1.2,
        label="T1 threshold ($10^{-6}\\sigma_{\\max}$)",
    )
    ax1.set_yscale("log")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8.5)
    ax1.set_ylabel("$\\max\\,\\mathrm{Re}\\,\\lambda(G) / \\sigma_{\\max}$ [-]")
    ax1.set_title("Generator Growth Rate ($\\alpha/2\\pi = 10$ MHz)")
    ax1.grid(True, axis="y", which="both", ls=":", alpha=0.6)
    ax1.legend(fontsize=8.5)

    # Panel 2: unstable eigenvector localisation for fine sigma_sampling
    loc_labels = [
        "Coarse block 0\n(EM energy)",
        "Patch block 1\n(EM energy)",
        "Interface faces\n(share of $E_b$)",
        "Interface edges\n(share of $E_e$)",
        "Depth Q1\n($u \\in [0, 0.25]$)",
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
    ax2.bar(x2, loc_vals, color=["C0", "C1", "C3", "C3", "C4", "C4"], width=0.58)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(loc_labels, fontsize=8)
    ax2.set_ylabel("Energy share [\\%]")
    ax2.set_ylim(0, 105)
    ax2.set_title("Unstable Eigenvector Localisation (`sigma_sampling='fine'`)")
    ax2.grid(True, axis="y", ls=":", alpha=0.6)

    fig.tight_layout()
    out_path = os.path.join(out_dir, "subgrid3d_cpml_patches_spectrum.png")
    fig.savefig(out_path, dpi=150)
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
