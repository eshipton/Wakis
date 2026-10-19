"""
Figures and tables for the subgrid arms on B2/B3, against every arm in the
shared results store (gs://.../bench-results, `wakis.benchmarks.results`).

    python plot_bench.py            # fetch, plot, write data/summary_B2_B3.md
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import os  # noqa: E402

from wakis.benchmarks.results import ROOT, fetch_records  # noqa: E402
from wakis.benchmarks.runner import cost_to_target  # noqa: E402

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
CASES = {
    "B2": (
        ["B2-Au-TE", "B2-Au-TM", "B2-Ag-TE", "B2-Ag-TM"],
        "q_ext_nrms",
        (0.01, 0.001),
    ),
    "B3": (["B3-g10", "B3-g5", "B3-g2"], "enh_peak", (0.10, 0.05)),
    "S1": (["S1-Si-g5", "S1-Si-g2"], "enh_peak", (0.05, 0.02)),
}
plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 150, "font.size": 8})


def work_eff(r):
    c = r["cost"]
    lts = c.get("lts") or {}
    p = lts.get("p", 1)
    # LTS: (p - 1) extra products on the fine set F per coarse step
    return c["unknowns"] * c["steps"] * (1 + (p - 1) * lts.get("fine_fraction", 0.0))


def style(arm):
    if arm.startswith("subgrid"):
        ls = ":" if "lts" in arm else "-"
        col = "#000000" if "gap" not in arm else "#7F7F7F"
        return dict(color=col, ls=ls, marker="P" if "gap" not in arm else "X", lw=1.6)
    if arm.startswith("tri"):
        return dict(
            color="#0072B2" if "graded" in arm else "#56B4E9", ls="-", marker="o"
        )
    if "conformal" in arm:
        return dict(
            color="#D55E00" if "graded" in arm else "#E69F00", ls="-", marker="s"
        )
    return dict(color="#CC79A7", ls="--", marker="v", alpha=0.6)


def load(case):
    root = os.environ.get("BENCH_ROOT", ROOT)  # a local mirror of the store
    recs = [
        r for r in fetch_records(case, root=root) if "metrics" in r and "error" not in r
    ]
    by = {}
    for r in recs:
        r["cost"]["work_eff"] = work_eff(r)
        by.setdefault(r["arm"], []).append(r)
    return by


def fig_spectra():
    """Best subgrid run vs the exact reference: Q_ext (B2-Au-TE) and the
    gap enhancement (B3-g5)."""
    from wakis.benchmarks import get_case

    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, case, key in ((axs[0], "B2-Au-TE", "q_ext"), (axs[1], "B3-g5", "enh")):
        c = get_case(case)
        lam = np.asarray(c.wavelengths) * 1e9
        ax.plot(lam, c.reference[key], "k-", lw=1.2, label="exact series" if key == "q_ext"
                else "exact T-matrix")  # fmt: skip
        by = load(case)
        for arm, mk in (("subgrid-2to1", "P"), ("subgrid-2to1-gap4", "X"),
                        ("tri-graded", "o"), ("cart-conformal-all-graded", "s")):  # fmt: skip
            rr = by.get(arm)
            if not rr:
                continue
            r = min(rr, key=lambda r: r["resolution"])
            sp = r["metrics"].get("spectra", {})
            if key in sp:
                ax.plot(lam[::3], np.asarray(sp[key])[::3], mk, ms=4, mfc="none",
                        color=style(arm)["color"],
                        label=f"Wakis FIT ({arm}), h = {r['resolution'] * 1e9:.2f} nm")  # fmt: skip
        ax.set_xlabel("vacuum wavelength [nm]")
        ax.set_ylabel("Q_ext" if key == "q_ext" else "|E/E0|^2 at the gap centre")
        ax.set_title(case)
        ax.legend(fontsize=6)
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "spectra_B2_B3.png")
    plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True)
    lines = []
    for phase, (cases, metric, targets) in CASES.items():
        fig, axs = plt.subplots(
            2, len(cases), figsize=(4.2 * len(cases), 7), squeeze=False
        )
        lines += [f"## {phase}: {metric}", "",
                  "| case | arm | " + " | ".join(
                      f"unknowns {100 * t:g} % | work {100 * t:g} % | time {100 * t:g} % [s]"
                      for t in targets) + " | best error (unknowns) |",
                  "|---|---|" + "---|---|---|" * len(targets) + "---|"]  # fmt: skip
        for j, case in enumerate(cases):
            by = load(case)
            for arm, rr in sorted(by.items()):
                rr = sorted(rr, key=lambda r: r["cost"]["unknowns"])
                e = [r["metrics"][metric] for r in rr]
                st = style(arm)
                axs[0, j].loglog(
                    [r["cost"]["unknowns"] for r in rr],
                    e,
                    label=f"Wakis FIT ({arm})",
                    ms=4,
                    **st,
                )
                axs[1, j].loglog([r["cost"]["work_eff"] for r in rr], e, ms=4, **st)
                cells = []
                for t in targets:
                    c = cost_to_target(rr, metric, t, ("unknowns", "work_eff", "run_s"))
                    cells += [
                        f"{c['unknowns']:,.0f}",
                        f"{c['work_eff']:.3g}",
                        f"{c['run_s']:.0f}",
                    ]
                k = int(np.nanargmin(e))
                lines.append(f"| {case} | {arm} | " + " | ".join(cells)
                             + f" | {100 * e[k]:.3g} % ({rr[k]['cost']['unknowns']:,}) |")  # fmt: skip
            for i in range(2):
                for t in targets:
                    axs[i, j].axhline(t, color="0.6", lw=0.6, ls="--")
                axs[i, j].grid(True, which="both", alpha=0.25)
            axs[0, j].set_title(case)
            axs[1, j].set_xlabel(
                "work = unknowns × steps (LTS: fine-set products added)"
            )
            axs[0, j].set_xlabel("field unknowns")
        axs[0, 0].set_ylabel(f"{metric} vs exact reference")
        axs[1, 0].set_ylabel(f"{metric} vs exact reference")
        axs[0, 0].legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(FIG / f"{phase}_error_vs_cost.png")
        plt.close(fig)
        lines.append("")
    (HERE / "data").mkdir(exist_ok=True)
    (HERE / "data" / "summary_B2_B3.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
    fig_spectra()
