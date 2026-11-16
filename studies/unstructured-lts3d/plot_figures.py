"""
Generate study figures for 3D tetrahedral local time stepping (`wakis.unstructured.lts3d`).

Figures:
1. `docs/img/lts3d_levels.png`: (a) setup schematic and mesh cut coloured by LTS level;
   (b) histogram of `dt_T / dt_global` with level thresholds and fine fractions.
2. `docs/img/lts3d_cost.png`: `err_td` vs wall time per simulated ns (log-log) across arms
   and named baselines, with equal-error annotations from the study JSON.
3. `docs/img/lts3d_stability.png`: `rho` and max level margin vs `dt / dt_global` with the
   derived global-limit line, modified energy drift for LTS and interface/buffer mutations,
   and the unstable run with conserved energy and growing state norm.

With `--quick`, reads `results/*_quick.json` and writes `results/*_quick.png` (untracked).
"""

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
IMG = os.path.join(ROOT, "docs", "img")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3})


def _load(name, quick=False):
    suffix = "_quick" if quick else ""
    path = os.path.join(RES, f"{name}{suffix}.json")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _out_path(stem, quick=False):
    if quick:
        os.makedirs(RES, exist_ok=True)
        return os.path.join(RES, f"{stem}_quick.png")
    os.makedirs(IMG, exist_ok=True)
    return os.path.join(IMG, f"{stem}.png")


def plot_levels(quick=False):
    from run_study import build_case_mesh

    from wakis.unstructured import UnstructuredFIT3D
    from wakis.unstructured.lts3d import element_levels, element_stability

    data = _load("bounds", quick=quick)["bounds"]
    rec = data[0]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))

    # Panel (a): mesh slice coloured by LTS level
    m, mats, _, _, _ = build_case_mesh(rec["case"], rec["size"], seed=rec["seed"])
    s = UnstructuredFIT3D(m, mats, hodge="voronoi")
    st = element_stability(s)
    lvl = element_levels(s, 2.2 * s.dt_max, [2, 2], [0.05, 0.10], stab=st)
    edge_global_lvl = np.zeros(m.n_edges, dtype=int)
    edge_global_lvl[s.free] = lvl
    cell_lvl = np.max(edge_global_lvl[m.cell_edges], axis=1)

    cen = m.nodes[m.cells].mean(axis=1)
    z_mid = 0.5 * (m.nodes[:, 2].min() + m.nodes[:, 2].max())
    dz = 1.2 * m.edge_length.mean()
    sel = np.abs(cen[:, 2] - z_mid) <= dz
    edges_slice = np.unique(m.cell_edges[sel].ravel())
    seg = m.nodes[m.edges[edges_slice]]
    axes[0].plot(seg[:, :, 0].T, seg[:, :, 1].T, color="#9CA3AF", lw=0.35, alpha=0.6)

    palette = {
        0: ("#2563EB", "Level 0 (coarse, dt)"),
        1: ("#F59E0B", "Level 1 (buffer/fine, dt/2)"),
        2: ("#DC2626", "Level 2 (finest, dt/4)"),
    }
    for lv, (col, lab) in palette.items():
        mask = sel & (cell_lvl == lv)
        if np.any(mask):
            axes[0].scatter(
                cen[mask, 0], cen[mask, 1], s=14, color=col, label=lab, zorder=3
            )
    axes[0].set_aspect("equal")
    axes[0].set_xlabel("x [m]")
    axes[0].set_ylabel("y [m]")
    axes[0].set_title("(a) Graded box mid-plane cut coloured by LTS level")
    axes[0].legend(loc="upper right", fontsize=8)

    # Panel (b): histogram of dt_T / dt_global with thresholds
    ratios_arr = np.asarray(rec["dt_T_over_dt_global"], dtype=float)
    ratios_arr = ratios_arr[np.isfinite(ratios_arr)]
    axes[1].hist(
        ratios_arr,
        bins=25,
        color="#3B82F6",
        edgecolor="black",
        alpha=0.75,
        label=r"Element bounds $\Delta t_T / \Delta t_{\mathrm{global}}$",
    )
    th1 = 1.1 / max(rec["theta_22"], 1e-12)
    th2 = 2.2 / max(rec["theta_22"], 1e-12)
    axes[1].axvline(th1, color="#DC2626", ls="--", lw=1.4, label="Level 2 threshold")
    axes[1].axvline(th2, color="#F59E0B", ls="-.", lw=1.4, label="Level 1 threshold")
    axes[1].set_xlabel(r"$\Delta t_T / \Delta t_{\mathrm{global}}$")
    axes[1].set_ylabel("Element count")
    axes[1].set_title(
        r"(b) Distribution of $\Delta t_T / \Delta t_{\mathrm{global}}$ and level cuts"
    )
    axes[1].legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    out = _out_path("lts3d_levels", quick=quick)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_cost(quick=False):
    acc = _load("accuracy", quick=quick)
    records = acc["records"]
    comps = acc.get("comparisons", {})

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    cases = sorted({r["case"] for r in records})[:2]
    if len(cases) == 1:
        cases = [cases[0], cases[0]]

    style = {
        "G": ("#1F2937", "o", "-"),
        "G-imp": ("#059669", "s", "-"),
        "G-ml3e-2": ("#7C3AED", "^", "--"),
        "G-ml1e-1": ("#A855F7", "v", "--"),
        "L": ("#DC2626", "D", "-"),
        "L-dt1.5": ("#F59E0B", "P", ":"),
        "L-imp": ("#10B981", "X", "-."),
        "GW": ("#6B7280", "o", "--"),
        "LW-edge_rate": ("#2563EB", "d", "-"),
    }

    for ax, case in zip(axes, cases):
        arms = sorted({r["arm"] for r in records if r["case"] == case})
        for arm in arms:
            pts = [
                r
                for r in records
                if r["case"] == case
                and r["arm"] == arm
                and r["seed"] == 1
                and np.isfinite(r["err_td"])
                and r["err_td"] > 0
            ]
            if not pts:
                continue
            pts.sort(key=lambda r: r["n_tets"])
            xs = [r["wall_per_ns"] for r in pts]
            ys = [r["err_td"] for r in pts]
            col, mk, ls = style.get(arm, ("#374151", ".", "-"))
            ax.loglog(xs, ys, marker=mk, ls=ls, color=col, label=arm, lw=1.3, ms=5)

        c_info = comps.get(case, {}).get("L_vs_G", {})
        status = c_info.get("status", "N/A")
        ax.set_xlabel("Wall time per simulated ns [s/ns]")
        ax.set_ylabel(
            r"Time-domain frequency error $|f_{\mathrm{td}}/f_{\mathrm{exact}} - 1|$"
        )
        ax.set_title(f"{case} (L vs G: {status})")
        ax.legend(fontsize=7.5, loc="best")

    fig.tight_layout()
    out = _out_path("lts3d_cost", quick=quick)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_stability(quick=False):
    stab = _load("stability", quick=quick)
    sweep = stab["sweep"]
    traces = stab["traces"]
    pred = stab["pred_global_limit_ratio"]

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0))

    # Panel (a): rho and max margin vs dt / dt_global
    xs = [r["dt_over_dt_global"] for r in sweep]
    rhos = [r["rho"] for r in sweep]
    margs = [r["max_margin"] for r in sweep]
    axes[0].plot(
        xs,
        rhos,
        "o-",
        color="#2563EB",
        lw=1.4,
        ms=4,
        label=r"$\rho = \frac{\Delta t}{2}\sqrt{\lambda_{\max}(S_0, M)}$",
    )
    axes[0].plot(
        xs,
        margs,
        "s--",
        color="#DC2626",
        lw=1.4,
        ms=4,
        label=r"$\max_l \tau_{l-1}^2 \mu_l / x_+$",
    )
    axes[0].axhline(1.0, color="black", lw=1.0, ls=":")
    axes[0].axvline(pred, color="#059669", lw=1.2, ls="-.", label="Derived limit (3.8)")
    axes[0].set_xlabel(r"$\Delta t / \Delta t_{\mathrm{global}}$")
    axes[0].set_ylabel("Stability indicator")
    axes[0].set_title(
        r"(a) $\rho$ and positivity margin vs $\Delta t / \Delta t_{\mathrm{global}}$"
    )
    axes[0].legend(fontsize=7.5, loc="upper left")

    # Panel (b): modified energy drift for stable LTS vs mutations
    for key, col, lab in (
        ("stable", "#059669", "Certified LTS (symmetric S_0)"),
        ("mut_interface", "#F59E0B", "Mutation: interface x1.01"),
        ("mut_drop_buffer", "#DC2626", "Mutation: drop_buffer"),
    ):
        d = np.maximum(np.asarray(traces[key]["drift"], dtype=float), 1e-17)
        axes[1].semilogy(np.arange(1, d.size + 1), d, color=col, lw=1.3, label=lab)
    axes[1].set_xlabel("Coarse step")
    axes[1].set_ylabel(r"$|E^{n+1/2} - E_0| / |E_0|$")
    axes[1].set_title("(b) Modified energy drift: LTS vs mutations")
    axes[1].legend(fontsize=7.5, loc="best")

    # Panel (c): unstable run with conserved energy and growing state norm
    u_drift = np.maximum(np.asarray(traces["unstable"]["drift"], dtype=float), 1e-17)
    u_norm = np.maximum(
        np.asarray(traces["unstable"]["norm_ratio"], dtype=float), 1e-17
    )
    steps_u = np.arange(1, u_drift.size + 1)
    axes[2].semilogy(
        steps_u,
        u_norm,
        "o-",
        color="#DC2626",
        lw=1.3,
        ms=3.5,
        label=r"State norm $\|u^n\| / \|u^0\|$",
    )
    axes[2].semilogy(
        steps_u,
        u_drift,
        "s--",
        color="#2563EB",
        lw=1.3,
        ms=3.5,
        label=r"Energy error $|E - E_0| / (|E_{\mathrm{kin}}| + |E_{\mathrm{pot}}|)$",
    )
    axes[2].set_xlabel("Coarse step (at 1.05 x dt_lts)")
    axes[2].set_ylabel("Dimensionless ratio")
    axes[2].set_title("(c) Unstable step: norm diverges, E conserved")
    axes[2].legend(fontsize=7.5, loc="best")

    fig.tight_layout()
    out = _out_path("lts3d_stability", quick=quick)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Plot from *_quick.json")
    args = parser.parse_args()

    p1 = plot_levels(quick=args.quick)
    p2 = plot_cost(quick=args.quick)
    p3 = plot_stability(quick=args.quick)
    print(f"plot_figures complete: {p1}, {p2}, {p3}")


if __name__ == "__main__":
    main()
