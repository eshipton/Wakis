"""Figures and summary table of the prism cavity study (reads data/*.json)."""

import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
FIG = os.path.join(HERE, "figures")
STYLE = {
    "prism": ("C0", "o", "Wakis FIT (prisms, uniform)"),
    "prism-graded": ("C0", "s", "Wakis FIT (prisms, graded)"),
    "cart": ("C3", "^", "Wakis FIT (Cartesian conformal, uniform)"),
    "cart-graded": ("C3", "v", "Wakis FIT (Cartesian conformal, graded)"),
    "tet": ("C2", "D", "Wakis FIT (tetrahedra, uniform)"),
    "tet-graded": ("C2", "P", "Wakis FIT (tetrahedra, graded)"),
}
TARGETS = [1e-2, 1e-3]


def load():
    rows = {}
    for fn in glob.glob(os.path.join(DATA, "*.json")):
        d = json.load(open(fn))
        if "arm" in d and np.isfinite(d.get("rms", np.nan)):
            rows.setdefault(d["arm"], []).append(d)
    for arm in rows:
        rows[arm].sort(key=lambda d: d["unknowns"])
    return rows


def to_target(rs, key, target):
    """log-log interpolation of `key` at rms = target (nan if not bracketed)."""
    e = np.array([r["rms"] for r in rs])
    v = np.array([float(r[key]) for r in rs])
    for i in range(len(rs) - 1):
        if e[i] >= target >= e[i + 1]:
            t = np.log(target / e[i]) / np.log(e[i + 1] / e[i])
            return float(np.exp(np.log(v[i]) + t * np.log(v[i + 1] / v[i])))
    return np.nan


def main():
    os.makedirs(FIG, exist_ok=True)
    rows = load()
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
    for arm, rs in rows.items():
        col, mk, lab = STYLE.get(arm, ("k", "x", arm))
        ls = "--" if "graded" in arm else "-"
        N = np.array([r["unknowns"] for r in rs])
        W = np.array([r["unknowns"] * r["steps"] for r in rs])
        Tt = np.array([r["run_s"] for r in rs])
        e = np.array([r["rms"] for r in rs])
        for ax, x in zip(axs, (N, W, Tt)):
            ax.loglog(x, 100 * e, ls, color=col, marker=mk, label=lab, mfc="none" if "graded" in arm else col)
    for ax, xl in zip(axs, ("field unknowns (E + B dofs)", "work = unknowns x steps", "uncontended run time [s] (1 core)")):
        ax.set_xlabel(xl)
        ax.set_ylabel("RMS error of 20 resonances [%]")
        ax.grid(True, which="both", alpha=0.3)
        for t in TARGETS:
            ax.axhline(100 * t, color="0.6", lw=0.8, ls=":")
    # second-order guide: error ~ N^(-2/3)
    x0 = axs[0].get_xlim()
    xs = np.geomspace(x0[0] * 2, x0[1] / 2, 10)
    axs[0].loglog(xs, 1.0 * (xs / xs[0]) ** (-2 / 3), "k:", lw=1, label="2nd order (h$^2$)")
    axs[0].legend(fontsize=7)
    fig.suptitle("PEC cylinder cavity with an n = 2 rod: 20 lowest resonances vs exact Bessel determinant")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "a_error_vs_cost.png"), dpi=150)

    # targets
    lines = ["| arm | target | unknowns | cells | work (unknowns x steps) | time [s] |", "|---|---|---|---|---|---|"]
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
    arms = [a for a in STYLE if a in rows]
    for j, t in enumerate(TARGETS):
        vals = {k: [to_target(rows[a], k, t) for a in arms] for k in ("unknowns", "cells")}
        for a in arms:
            rs = rows[a]
            for r in rs:
                r["work"] = r["unknowns"] * r["steps"]
        vals["work"] = [to_target(rows[a], "work", t) for a in arms]
        vals["run_s"] = [to_target(rows[a], "run_s", t) for a in arms]
        for i, a in enumerate(arms):
            lines.append(
                f"| {a} | {100 * t:g} % | {vals['unknowns'][i]:,.0f} | {vals['cells'][i]:,.0f} | "
                f"{vals['work'][i]:.3g} | {vals['run_s'][i]:.0f} |"
            )
        for ax, k in zip(axs, ("unknowns", "work", "run_s")):
            x = np.arange(len(arms)) + 0.4 * j - 0.2
            ax.bar(x, vals[k], 0.38, color=[STYLE[a][0] for a in arms], alpha=0.45 + 0.5 * j,
                   label=f"{100 * t:g} %")
    for ax, k in zip(axs, ("unknowns", "work (unknowns x steps)", "uncontended time [s]")):
        ax.set_xticks(range(len(arms)))
        ax.set_xticklabels(arms, rotation=30, fontsize=8)
        ax.set_yscale("log")
        ax.set_ylabel(k)
        ax.legend(title="RMS target", fontsize=8)
        ax.grid(True, axis="y", which="both", alpha=0.3)
    fig.suptitle("Cost to reach a target RMS resonance error (log-log interpolation)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "b_cost_to_target.png"), dpi=150)

    # per-run table
    out = ["| arm | res | unknowns | cells | dt [as] | steps | run [s] | ns/(unknown step) | RMS err | RMS spatial | max err |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for a in arms:
        for r in sorted(rows[a], key=lambda r: r["res"]):
            out.append(
                f"| {a} | {r['res']:g} | {r['unknowns']:,} | {r['cells']:,} | {r['dt'] * 1e18:.1f} | {r['steps']} | "
                f"{r['run_s']:.1f} | {1e9 * r['run_s'] / (r['unknowns'] * r['steps']):.2f} | {100 * r['rms']:.3f} % | "
                f"{100 * r['rms_spatial']:.3f} % | {100 * r['max']:.3f} % |"
            )
    with open(os.path.join(DATA, "summary.md"), "w") as fh:
        fh.write("\n".join(out) + "\n\nCost to reach a target RMS error (log-log interpolation):\n\n" + "\n".join(lines) + "\n")
    print("\n".join(out))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
