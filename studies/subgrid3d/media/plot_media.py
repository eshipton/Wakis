# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Plot figures for the 3D subgrid dispersive and subpixel media study
(`studies/subgrid3d/media/run_media.py`).

Produces:
  - `subgrid3d_media_slab_convergence.png`:
    (a) Setup schematic: dispersive / conformal slab crossing a 2:1 subgrid
        patch in the (x, z) cross-section (y = b/2), showing coarse cells (H),
        2:1 fine cells (h = H/2), hanging-edge patch boundary, and slab 0 < z < z_s.
    (b) Dispersive ADE convergence vs fine cell step h_z [m] for Lorentz and
        lossless Drude slabs crossing the 2:1 patch vs the 1D shooting reference
        and vs uniform fine `SubgridFIT3D` (without patch, == `SolverFIT3D`),
        plus the `mutate="lump_no_rowsum"` mutation and O(h^2) guide.
    (c) Subpixel Kottke vs staircase vs `mutate="swap_projection"` relative
        frequency error vs fine cell step h_z [m] for a non-node-plane
        dielectric slab (eps_r = 4, z_s = 0.361 c) crossing a 2:1 patch.
  - `subgrid3d_media_dispersive_cube.png`:
    (a) Localized Lorentz & Drude cube: resonance error vs state unknowns [DOF]
    (b) Localized Lorentz & Drude cube: resonance error vs work per ns [DOF/ns]
    (c) Honest panel (exterior-limited cavity, patch interface on dielectric
        jump): resonance error vs work per ns [DOF/ns] where the 2:1 patch does
        not beat the uniform fine baseline (`SubgridFIT3D` == `SolverFIT3D`).
  - `subgrid3d_media_subpixel_sphere.png`:
    (a) Dielectric sphere (eps_r = 4) inside a 2:1 patch: mean relative
        resonance error vs fine cell step h [m] for staircase and subpixel
        (`n_sub = 2, 4, 6`, moment normal vs exact radial normal).
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
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

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


def plot_slab_convergence(slab, out_dir):
    d6 = slab["d6_dispersive"]
    c2 = slab["c2_subpixel"]
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.5), constrained_layout=True)

    # Panel (a): Labelled (x, z) setup & mesh schematic at N_z = 9
    ax = axes[0]
    a, _b, c = d6["box_m"]
    z_s = d6["z_s_m"]
    (xp0, xp1), _yp, (zp0, zp1) = d6["patch_bounds_m"]
    nx0, nz0 = 3, 9
    xs = np.linspace(0.0, a, nx0 + 1)
    zs = np.linspace(0.0, c, nz0 + 1)

    # Vacuum region and dispersive/dielectric slab region
    ax.add_patch(
        Rectangle(
            (0.0, z_s),
            a,
            c - z_s,
            facecolor="#f4f6f9",
            edgecolor="none",
            zorder=1,
        )
    )
    ax.add_patch(
        Rectangle(
            (0.0, 0.0),
            a,
            z_s,
            facecolor="#d0e4f7",
            edgecolor="none",
            alpha=0.85,
            zorder=1,
            label="Dispersive / dielectric slab ($0 < z < z_s$)",
        )
    )

    # Coarse grid lines (step H)
    for xv in xs:
        ax.plot([xv, xv], [0.0, c], color="#7f7f7f", lw=0.9, zorder=2)
    for zv in zs:
        ax.plot([0.0, a], [zv, zv], color="#7f7f7f", lw=0.9, zorder=2)

    # 2:1 fine subgrid patch lines (step h = H / 2)
    xf_p = np.linspace(xp0, xp1, 2 * 1 + 1)
    zf_p = np.linspace(zp0, zp1, 2 * 4 + 1)
    for xv in xf_p:
        ax.plot([xv, xv], [zp0, zp1], color="#1f77b4", lw=1.1, zorder=3)
    for zv in zf_p:
        ax.plot([xp0, xp1], [zv, zv], color="#1f77b4", lw=1.1, zorder=3)

    # Highlight hanging-edge 2:1 patch boundary
    ax.add_patch(
        Rectangle(
            (xp0, zp0),
            xp1 - xp0,
            zp1 - zp0,
            facecolor="none",
            edgecolor="#d62728",
            lw=2.0,
            ls="-",
            zorder=4,
            label="2:1 patch interface ($M_k = |P|^T(m_{p,k}\\circ|P|\\mathbf{1})$)",
        )
    )

    # Material interface z = z_s (D6 on node plane z_s=0.30 m; C2 off-plane z_s=0.361 c)
    ax.axhline(
        z_s,
        color="#0b559f",
        lw=1.8,
        ls="--",
        zorder=5,
        label=f"D6 slab top $z_s = {z_s:.2f}$ m",
    )
    z_s_c2_scaled = c2["r_frac"] * c
    ax.axhline(
        z_s_c2_scaled,
        color="#2ca02c",
        lw=1.6,
        ls="-.",
        zorder=5,
        label=f"C2 sub-cell cut $z_s = {c2['r_frac']:.3f}\\,c$",
    )

    # Outer PEC cavity boundary
    ax.add_patch(
        Rectangle(
            (0.0, 0.0),
            a,
            c,
            facecolor="none",
            edgecolor="#111111",
            lw=2.0,
            zorder=6,
        )
    )
    ax.text(
        0.5 * a,
        0.045,
        "Slab $\\varepsilon(\\omega)$ or $\\varepsilon_r=4$",
        ha="center",
        va="bottom",
        fontsize=8.2,
        fontweight="bold",
        color="#08306b",
        zorder=7,
    )
    ax.text(
        0.5 * a,
        0.34,
        "2:1 fine patch\n($h_z = H_z / 2$)",
        ha="center",
        va="center",
        fontsize=8.0,
        color="#1f77b4",
        bbox={
            "boxstyle": "round,pad=0.2",
            "fc": "white",
            "ec": "#1f77b4",
            "alpha": 0.88,
        },
        zorder=7,
    )
    ax.text(
        0.5 * a,
        0.56,
        "Coarse vacuum ($H_z = c / N_z$)",
        ha="center",
        va="center",
        fontsize=8.2,
        color="#333333",
        zorder=7,
    )
    ax.set_xlim(-0.02 * a, 1.02 * a)
    ax.set_ylim(-0.02 * c, 1.02 * c)
    ax.set_xlabel("Transverse coordinate $x$ [m]")
    ax.set_ylabel("Axial coordinate $z$ [m]")
    ax.set_title("(a) Setup: slab crossing 2:1 patch ($x$–$z$ cut)")
    ax.legend(fontsize=7.1, loc="upper right", framealpha=0.92)

    # Panel (b): D6 convergence vs h_z for Lorentz and Drude slabs
    ax = axes[1]
    hz_d6 = np.array(d6["hz_fine_m"], float)
    ax.loglog(
        hz_d6,
        d6["err_lor_1d"],
        "o-",
        color="#1f77b4",
        lw=1.8,
        label=f"Lorentz 2:1 patch vs 1D shooting (slope={d6['ls_ord_lor_1d']:.2f})",
    )
    ax.loglog(
        hz_d6,
        d6["err_lor_fine"],
        "s--",
        color="#1f77b4",
        lw=1.5,
        alpha=0.85,
        label=f"Lorentz 2:1 patch vs uniform fine SubgridFIT3D (slope={d6['ls_ord_lor_fine']:.2f})",
    )
    ax.loglog(
        hz_d6,
        d6["err_dru_1d"],
        "^-",
        color="#d62728",
        lw=1.8,
        label=f"Drude 2:1 patch vs 1D shooting (slope={d6['ls_ord_dru_1d']:.2f})",
    )
    ax.loglog(
        hz_d6,
        d6["err_dru_fine"],
        "v--",
        color="#d62728",
        lw=1.5,
        alpha=0.85,
        label=f"Drude 2:1 patch vs uniform fine SubgridFIT3D (slope={d6['ls_ord_dru_fine']:.2f})",
    )
    ax.loglog(
        hz_d6,
        d6["err_lor_mut"],
        "x:",
        color="#1f77b4",
        lw=1.3,
        alpha=0.75,
        label="Lorentz unlumped mutation (lump_no_rowsum)",
    )
    ax.loglog(
        hz_d6,
        d6["err_dru_mut"],
        "+:",
        color="#d62728",
        lw=1.3,
        alpha=0.75,
        label="Drude unlumped mutation (lump_no_rowsum)",
    )
    # O(h^2) and O(h) guides
    h_ref = hz_d6
    ax.loglog(
        h_ref,
        0.75 * d6["err_dru_1d"][-1] * (h_ref / h_ref[-1]) ** 2.0,
        "k--",
        lw=1.1,
        alpha=0.6,
        label="$\\mathcal{O}(h_z^2)$ guide",
    )
    ax.loglog(
        h_ref,
        1.15 * d6["err_lor_mut"][-1] * (h_ref / h_ref[-1]) ** 1.0,
        "k:",
        lw=1.0,
        alpha=0.5,
        label="$\\mathcal{O}(h_z)$ guide",
    )
    ax.set_xlabel("Fine patch axial cell step $h_z = H_z / 2$ [m]")
    ax.set_ylabel("Mean relative frequency error (3 modes) [1]")
    ax.set_title("(b) Drude & Lorentz slab convergence across 2:1 patch")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=6.9, loc="best")

    # Panel (c): C2 subpixel vs staircase error vs h_z for eps_r = 4 non-node-plane slab
    ax = axes[2]
    hz_c2 = np.array(c2["hz_fine_m"], float)
    ax.loglog(
        hz_c2,
        c2["err_sub"],
        "^-",
        color="#1f77b4",
        lw=1.8,
        label=f"Subpixel Kottke (averaging='subpixel', slope={c2['ls_sub']:.2f})",
    )
    ax.loglog(
        hz_c2,
        c2["err_st"],
        "o-",
        color="#7f7f7f",
        lw=1.6,
        label=f"Staircase (averaging='staircase', slope={c2['ls_st']:.2f})",
    )
    ax.loglog(
        hz_c2,
        c2["err_mut"],
        "d--",
        color="#ff7f0e",
        lw=1.5,
        label=f"Swapped projection (swap_projection, slope={c2['ls_mut']:.2f})",
    )
    ax.loglog(
        hz_c2,
        0.75 * c2["err_sub"][-1] * (hz_c2 / hz_c2[-1]) ** 2.0,
        "k--",
        lw=1.1,
        alpha=0.6,
        label="$\\mathcal{O}(h_z^2)$ guide",
    )
    ax.loglog(
        hz_c2,
        1.1 * c2["err_st"][-1] * (hz_c2 / hz_c2[-1]) ** 1.0,
        "k:",
        lw=1.0,
        alpha=0.5,
        label="$\\mathcal{O}(h_z)$ guide",
    )
    ax.set_xlabel("Fine patch axial cell step $h_z = H_z / 2$ [m]")
    ax.set_ylabel(
        "Mean relative frequency error $|f - f_{\\mathrm{1D}}| / f_{\\mathrm{1D}}$ [1]"
    )
    ax.set_title(
        "(c) Subpixel vs staircase slab ($\\varepsilon_r = 4$, $z_s = 0.361\\,c$)"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7.3, loc="best")

    out_path = os.path.join(out_dir, "subgrid3d_media_slab_convergence.png")
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def plot_dispersive_cube(disp, out_dir):
    rows = disp["rows"]
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.3), constrained_layout=True)

    arm_styles = {
        "uniform_fine": {
            "marker": "o",
            "ls": "-",
            "label": "Uniform fine h (SubgridFIT3D == SolverFIT3D)",
        },
        "patch_r2": {
            "marker": "s",
            "ls": "--",
            "label": "2:1 patch over cube (fine h)",
        },
        "uniform_coarse": {
            "marker": "^",
            "ls": ":",
            "label": "Uniform coarse H = 2h",
        },
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
    ax.set_title("(a) Localized cube (3.3% vol.): error vs unknowns")
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
    ax.set_title("(c) Honest regime (20% block on jump)")
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
        "staircase": {
            "color": "#7f7f7f",
            "marker": "o",
            "ls": "-",
            "label": f"Staircase (n_sub=1, slope={orders['staircase']:.2f})",
        },
        "subpixel_nsub2_momentN": {
            "color": "#ff7f0e",
            "marker": "s",
            "ls": "--",
            "label": "Subpixel Kottke (n_sub=2, moment normal)",
        },
        "subpixel_nsub4_momentN": {
            "color": "#1f77b4",
            "marker": "^",
            "ls": "-",
            "label": "Subpixel Kottke (n_sub=4, moment normal)",
        },
        "subpixel_nsub6_momentN": {
            "color": "#2ca02c",
            "marker": "v",
            "ls": "-.",
            "label": "Subpixel Kottke (n_sub=6, moment normal)",
        },
        "subpixel_nsub4_exactN": {
            "color": "#9467bd",
            "marker": "d",
            "ls": ":",
            "label": "Subpixel Kottke (n_sub=4, exact radial normal)",
        },
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
    written = []
    if "slab" in data:
        written.append(plot_slab_convergence(data["slab"], args.out_dir))
    written.append(plot_dispersive_cube(data["dispersive"], args.out_dir))
    written.append(plot_subpixel_sphere(data["sphere"], args.out_dir))
    print("Wrote " + ", ".join(written))


if __name__ == "__main__":
    main()
