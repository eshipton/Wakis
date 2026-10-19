"""
Figures and tables of the polygonal-FIT arms on B2/B3 (and the baselines,
when their records are present).

    python plot_bench.py [--base DIR ...]

Reads data/*.json (this study) and any extra record folders (baselines from
../benchmarks/bench_study.py). Writes figures/ and data/summary.md.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
STYLE = {
    "poly-hybrid": dict(color="#009E73", marker="P", label="Wakis FIT (hybrid squares + triangles, gap-refined)"),
    "poly-voronoi": dict(color="#CC79A7", marker="h", label="Wakis FIT (Voronoi polygons)"),
    "tri-graded": dict(color="#0072B2", marker="o", label="Wakis FIT (triangles graded)"),
    "tri-uniform": dict(color="#56B4E9", marker="o", label="Wakis FIT (triangles uniform)"),
    "cart-conformal-all-graded": dict(color="#D55E00", marker="s", label="Wakis FIT (Cartesian conformal graded)"),
    "cart-conformal-all-uniform": dict(color="#E69F00", marker="s", label="Wakis FIT (Cartesian conformal uniform)"),
    "cart-staircase-graded": dict(color="#999999", marker="x", label="Wakis FIT (Cartesian staircase graded)"),
}  # fmt: skip
CASES = ["B2-Au-TE", "B2-Au-TM", "B2-Ag-TE", "B2-Ag-TM", "B3-g10", "B3-g5", "B3-g2"]


def load(folders):
    recs = []
    for d in folders:
        for f in sorted(Path(d).glob("**/*.json")):
            try:
                r = json.loads(f.read_text())
            except Exception:
                continue
            if "case" in r and "metrics" in r:
                recs.append(r)
    return recs


def primary(case):
    return "enh_peak" if case.startswith("B3") else "q_ext_nrms"


def series(recs, case, arm, metric):
    rr = [r for r in recs if r["case"] == case and r["arm"] == arm]
    rr.sort(key=lambda r: -r["resolution"])
    x = np.array([r["cost"]["unknowns"] * r["cost"]["steps"] for r in rr], float)
    u = np.array([r["cost"]["unknowns"] for r in rr], float)
    t = np.array([r["cost"]["run_s"] for r in rr], float)
    y = np.array([abs(r["metrics"].get(metric, np.nan)) for r in rr], float)
    h = np.array([r["resolution"] for r in rr], float)
    return h, u, x, t, y


def to_target(cost, err, target):
    """Cost where the error falls below `target` for good: log-log
    interpolation between the last rung above the target and the next one
    (robust to non-monotone ladders); the coarsest rung if all are below."""
    cost, err = np.asarray(cost, float), np.asarray(err, float)
    above = np.flatnonzero(err > target)
    if not above.size:
        return float(cost[0]) if err.size else np.nan
    i = above[-1]
    if i + 1 >= len(err):
        return np.nan
    s = (np.log(target) - np.log(err[i])) / (np.log(err[i + 1]) - np.log(err[i]))
    return float(np.exp(np.log(cost[i]) + s * (np.log(cost[i + 1]) - np.log(cost[i]))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", nargs="*", default=[])
    a = ap.parse_args()
    recs = load([HERE / "data", *a.base])
    arms = [k for k in STYLE if any(r["arm"] == k for r in recs)]
    (HERE / "figures").mkdir(exist_ok=True)
    fig, axs = plt.subplots(2, 4, figsize=(15, 7))
    lines = ["| case | arm | h [nm] | unknowns | steps | work | run [s] | Q_ext NRMS | enh. peak err | λ_peak err |",
             "|---|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    tgt = ["", "Cost to target (log-log interpolation; primary metric: Q_ext NRMS for B2, gap enhancement peak for B3):", "",
           "| case | arm | target | unknowns | work | run [s] |", "|---|---|---|---|---|---|"]  # fmt: skip
    for ax, case in zip(axs.ravel(), CASES):
        met = primary(case)
        for arm in arms:
            h, u, w, t, y = series(recs, case, arm, met)
            if not len(h):
                continue
            st = STYLE[arm]
            ax.loglog(w, 100 * y, marker=st["marker"], color=st["color"], ms=4, lw=1, label=st["label"])
            for r in sorted([r for r in recs if r["case"] == case and r["arm"] == arm], key=lambda r: -r["resolution"]):
                m, c = r["metrics"], r["cost"]
                lines.append(
                    f"| {case} | {arm} | {r['resolution'] * 1e9:.2f} | {c['unknowns']:,} | {c['steps']} | {c['unknowns'] * c['steps']:.3g} | {c['run_s']:.0f} | "
                    f"{100 * m['q_ext_nrms']:.2f} % | {100 * abs(m.get('enh_peak', np.nan)):.2f} % | {100 * abs(m['q_ext_lam_peak']):.2f} % |"
                )
            targets = (0.05, 0.02) if case.startswith("B3") else (0.01, 0.001)
            for T in targets:
                tgt.append(f"| {case} | {arm} | {100 * T:g} % | {to_target(u, y, T):,.0f} | {to_target(w, y, T):.3g} | {to_target(t, y, T):,.0f} |")
        ax.set_title(f"{case}: {met}", fontsize=8)
        ax.set_xlabel("work = unknowns × steps")
        ax.set_ylabel("error [%]")
        ax.grid(True, which="both", alpha=0.2)
    axs.ravel()[-1].axis("off")
    h_, l_ = axs.ravel()[0].get_legend_handles_labels()
    axs.ravel()[-1].legend(h_, l_, fontsize=7, loc="center")
    fig.tight_layout()
    fig.savefig(HERE / "figures" / "a_error_vs_work.png", dpi=150)
    plt.close(fig)
    # spectra at a mid resolution vs the exact reference
    from wakis.benchmarks import get_case

    fig, axs = plt.subplots(1, 3, figsize=(14, 6.2))
    for ax, case in zip(axs, ("B2-Au-TE", "B2-Ag-TE", "B3-g5")):
        c = get_case(case)
        lam = np.asarray(c.wavelengths) * 1e9
        key = "enh" if case.startswith("B3") else "q_ext"
        ax.plot(lam, c.reference[key], "k-", lw=1, label="exact (T-matrix / series)")
        for arm in arms:
            rr = [r for r in recs if r["case"] == case and r["arm"] == arm]
            if not rr:
                continue
            r = min(rr, key=lambda r: abs(r["resolution"] - 2e-9))
            st = STYLE[arm]
            ax.plot(lam[::3], np.asarray(r["metrics"]["spectra"][key])[::3], st["marker"], color=st["color"],
                    mfc="none", ms=4, label=f"{st['label']}, h = {r['resolution'] * 1e9:.1f} nm, {r['cost']['unknowns']:,} unknowns")  # fmt: skip
        ax.set_xlabel("wavelength [nm]")
        ax.set_ylabel("|E/E0|² at gap centre" if key == "enh" else "Q_ext")
        ax.set_title(case, fontsize=8)
        ax.legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    fig.tight_layout()
    fig.savefig(HERE / "figures" / "b_spectra.png", dpi=150)
    plt.close(fig)
    (HERE / "data").mkdir(exist_ok=True)
    (HERE / "data" / "summary.md").write_text("\n".join(lines + tgt) + "\n")


if __name__ == "__main__":
    main()
