# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Plot figures for the 3D subgrid dispersive and subpixel media study
(`studies/subgrid3d/media/run_media.py`).

Produces:
  - `subgrid3d_media_dispersive_cube.png`:
    (a) Localized Lorentz & Drude cube: resonance error vs state unknowns [1]
    (b) Localized Lorentz & Drude cube: resonance error vs work per ns [DOF/ns]
    (c) Honest panel (over-patched exterior-limited cavity): resonance error
        vs work per ns [DOF/ns] where the 2:1 patch does not beat the uniform
        baseline (`SubgridFIT3D` without patch at fine h).
  - `subgrid3d_media_subpixel_sphere.png`:
    (a) Dielectric sphere (eps_r = 4) inside a 2:1 patch: mean relative
        resonance error vs fine cell step h [m] for staircase and subpixel
        (`n_sub = 2, 4, ...`).
    (b) Per-mode relative frequency error vs coarse grid count N_x [cells].

Usage
-----
    python studies/subgrid3d/media/plot_media.py --in DIR --out DIR
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "docs", "img"))


def _load_results(in_dir):
    if os.path.isfile(in_dir):
        with open(in_dir) as f:
            return json.load(f)
    for fname in ("media.json", "media_quick.json"):
        candidate = os.path.join(in_dir, fname)
        if os.path.exists(candidate):
            with open(candidate) as f:
                return json.load(f)
    raise FileNotFoundError(
        f"Neither media.json nor media_quick.json found in {in_dir}"
    )


def plot_dispersive_cube(disp, out_dir):
    rows = disp["rows"]
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.3), constrained_layout=True)

    arm_styles = {
        "uniform_fine": dict(
            marker="o",
            ls="-",
            label="Uniform fine h (SubgridFIT3D == SolverFIT3D)",
        ),
        "patch_r2": dict(
            marker="s",
            ls="--",
            label="2:1 patch over cube (fine h)",
        ),
        "uniform_coarse": dict(
            marker="^",
            ls=":",
            label="Uniform coarse H = 2h",
        ),
    }
    model_colors = {"lorentz": "#1f77b4", "drude": "#d62728"}

    # Panel (a): Localized cube — error vs state unknowns n_state
    ax = axes[0]
    for model in ("lorentz", "drude"):
        for arm, st in arm_styles.items():
            sub = sorted(
                [
                    r
                    for r in rows
                    if r["case"] == "localized"
                    and r["model"] == model
                    and r["arm"] == arm
                ],
                key=lambda x: x["nx_coarse"],
            )
            if not sub:
                continue
            xs = [r["n_state"] for r in sub]
            ys = [max(r["rel_err_ref"], 1e-7) for r in sub]
            ax.loglog(
                xs,
                ys,
                color=model_colors[model],
                marker=st["marker"],
                ls=st["ls"],
                lw=1.6,
                label=f"{model.capitalize()} — {st['label']}",
            )
    ax.set_xlabel("State unknowns $n_{\\mathrm{state}}$ [DOF]")
    ax.set_ylabel("Relative frequency error $|f_1 - f_*| / f_*$ [1]")
    ax.set_title("(a) Localized cube (5.6% vol.): error vs unknowns")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7.2, loc="best")

    # Panel (b): Localized cube — error vs work per ns (n_state / dt)
    ax = axes[1]
    for model in ("lorentz", "drude"):
        for arm in ("uniform_fine", "patch_r2"):
            st = arm_styles[arm]
            sub = sorted(
                [
                    r
                    for r in rows
                    if r["case"] == "localized"
                    and r["model"] == model
                    and r["arm"] == arm
                ],
                key=lambda x: x["nx_coarse"],
            )
            if not sub:
                continue
            xs = [r["work_per_ns"] for r in sub]
            ys = [max(r["rel_err_ref"], 1e-7) for r in sub]
            ax.loglog(
                xs,
                ys,
                color=model_colors[model],
                marker=st["marker"],
                ls=st["ls"],
                lw=1.6,
                label=f"{model.capitalize()} — {st['label']}",
            )
    ax.set_xlabel("Work per nanosecond $n_{\\mathrm{state}} / \\Delta t$ [DOF / ns]")
    ax.set_ylabel("Relative frequency error $|f_1 - f_*| / f_*$ [1]")
    ax.set_title("(b) Localized cube: error vs time-domain work")
    ax.grid(True, which="both", alpha=0.3)

    # Panel (c): Honest panel — over-patched exterior-limited cavity
    ax = axes[2]
    for model in ("lorentz", "drude"):
        for arm in ("uniform_fine", "patch_r2", "uniform_coarse"):
            st = arm_styles[arm]
            sub = sorted(
                [
                    r
                    for r in rows
                    if r["case"] == "exterior_limited"
                    and r["model"] == model
                    and r["arm"] == arm
                ],
                key=lambda x: x["nx_coarse"],
            )
            if not sub:
                continue
            xs = [r["work_per_ns"] for r in sub]
            ys = [max(r["rel_err_ref"], 1e-7) for r in sub]
            ax.loglog(
                xs,
                ys,
                color=model_colors[model],
                marker=st["marker"],
                ls=st["ls"],
                lw=1.6,
                label=f"{model.capitalize()} — {st['label']}",
            )
    ax.set_xlabel("Work per nanosecond $n_{\\mathrm{state}} / \\Delta t$ [DOF / ns]")
    ax.set_ylabel("Relative frequency error $|f_1 - f_*| / f_*$ [1]")
    ax.set_title("(c) Honest regime (53% patch, exterior-dominated)")
    ax.grid(True, which="both", alpha=0.3)

    out_path = os.path.join(out_dir, "subgrid3d_media_dispersive_cube.png")
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def plot_subpixel_sphere(sphere, out_dir):
    rows = sphere["rows"]
    orders = sphere["orders"]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), constrained_layout=True)

    style_map = {
        "staircase": dict(
            color="#7f7f7f",
            marker="o",
            ls="-",
            label=f"Staircase (n_sub=1, slope={orders['staircase']:.2f})",
        ),
        "subpixel_nsub2_momentN": dict(
            color="#ff7f0e",
            marker="s",
            ls="--",
            label="Subpixel Kottke (n_sub=2, moment normal)",
        ),
        "subpixel_nsub4_momentN": dict(
            color="#1f77b4",
            marker="^",
            ls="-",
            label="Subpixel Kottke (n_sub=4, moment normal)",
        ),
        "subpixel_nsub6_momentN": dict(
            color="#2ca02c",
            marker="v",
            ls="-.",
            label="Subpixel Kottke (n_sub=6, moment normal)",
        ),
        "subpixel_nsub4_exactN": dict(
            color="#9467bd",
            marker="d",
            ls=":",
            label="Subpixel Kottke (n_sub=4, exact radial normal)",
        ),
    }

    # Panel (a): Mean relative error across 4 modes vs fine cell step h [m]
    ax = axes[0]
    for label, st in style_map.items():
        sub = sorted(
            [r for r in rows if r["label"] == label], key=lambda x: x["h_fine_m"]
        )
        if not sub:
            continue
        hs = np.array([r["h_fine_m"] for r in sub], float)
        es = np.array([max(r["rel_err_mean"], 1e-7) for r in sub], float)
        ax.loglog(
            hs,
            es,
            color=st["color"],
            marker=st["marker"],
            ls=st["ls"],
            lw=1.7,
            label=st["label"],
        )

    # Reference O(h) and O(h^2) guides
    st_rows = sorted(
        [r for r in rows if r["label"] == "staircase"], key=lambda x: x["h_fine_m"]
    )
    h_guide = np.array([r["h_fine_m"] for r in st_rows], float)
    ax.loglog(
        h_guide,
        st_rows[-1]["rel_err_mean"] * (h_guide / h_guide[-1]) ** 1.0,
        "k--",
        lw=1.0,
        alpha=0.5,
        label="$\\mathcal{O}(h)$ reference",
    )
    ax.loglog(
        h_guide,
        0.5 * st_rows[-1]["rel_err_mean"] * (h_guide / h_guide[-1]) ** 2.0,
        "k:",
        lw=1.0,
        alpha=0.5,
        label="$\\mathcal{O}(h^2)$ reference",
    )
    ax.set_xlabel("Fine patch cell size $h$ [m]")
    ax.set_ylabel("Mean relative frequency error (4 modes) [1]")
    ax.set_title("(a) Dielectric sphere ($\\varepsilon_r = 4$) in 2:1 patch")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7.8, loc="best")

    # Panel (b): Individual mode relative errors vs N_x for staircase vs subpixel (n_sub=4)
    ax = axes[1]
    for label, ls_style, col in [
        ("staircase", "--", "#7f7f7f"),
        ("subpixel_nsub4_momentN", "-", "#1f77b4"),
    ]:
        sub = sorted(
            [r for r in rows if r["label"] == label], key=lambda x: x["nx_coarse"]
        )
        nxs = [r["nx_coarse"] for r in sub]
        for m_idx in range(4):
            ys = [max(r["rel_err_modes"][m_idx], 1e-7) for r in sub]
            lbl = (
                f"{'Staircase' if label == 'staircase' else 'Subpixel (n_sub=4)'} "
                f"(modes 1–4)"
                if m_idx == 0
                else None
            )
            ax.semilogy(
                nxs,
                ys,
                ls=ls_style,
                marker="o" if label == "staircase" else "^",
                color=col,
                alpha=0.45 + 0.15 * m_idx,
                lw=1.3,
                label=lbl,
            )
    ax.set_xlabel("Coarse grid count $N_x$ [cells]")
    ax.set_ylabel("Per-mode relative frequency error $|f_m - f_{m,*}| / f_{m,*}$ [1]")
    ax.set_title("(b) Mode-by-mode error vs coarse grid count $N_x$")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8.2, loc="best")

    out_path = os.path.join(out_dir, "subgrid3d_media_subpixel_sphere.png")
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--in", dest="in_dir", default=HERE, help="Input directory or JSON file"
    )
    ap.add_argument(
        "--out", dest="out_dir", default=DEFAULT_OUT, help="Output directory for PNGs"
    )
    args = ap.parse_args()

    data = _load_results(args.in_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    p1 = plot_dispersive_cube(data["dispersive"], args.out_dir)
    p2 = plot_subpixel_sphere(data["sphere"], args.out_dir)
    print(f"Wrote {p1} and {p2}")


if __name__ == "__main__":
    main()
