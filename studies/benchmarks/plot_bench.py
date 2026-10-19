"""Error vs unknowns / work and cost-to-target tables of the benchmark sweeps.

``python bench_study.py plot --phase B2`` reads data/<phase>/*.json and writes
figures/<phase>_error.png and data/<phase>_summary.md.
"""

import glob
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from wakis.benchmarks import runner  # noqa: E402

HERE = Path(__file__).resolve().parent
STYLE = {
    "cart-staircase-uniform": dict(color="#D55E00", marker="s", ls="--"),
    "cart-staircase-graded": dict(color="#D55E00", marker="s", ls="-"),
    "cart-conformal-all-uniform": dict(color="#E69F00", marker="D", ls="--"),
    "cart-conformal-all-graded": dict(color="#E69F00", marker="D", ls="-"),
    "tri-uniform": dict(color="#56B4E9", marker="o", ls="--"),
    "tri-graded": dict(color="#0072B2", marker="o", ls="-"),
}
METRICS = {
    "B2": [("q_ext_nrms", (0.02, 0.01)), ("q_ext_lam_peak", (2e-3, 1e-3))],
    "B3": [("q_ext_nrms", (0.02, 0.01)), ("enh_nrms", (0.1, 0.05))],
}


def load(phase):
    recs = defaultdict(list)
    bad = []
    for f in glob.glob(str(HERE / "data" / phase / "*.json")):
        r = runner.load_records([f])[0]
        if "error" in r:
            bad.append((Path(f).name, r["error"]))
            continue
        recs[(r["case"], r["arm"])].append(r)
    for k in recs:
        recs[k].sort(key=lambda r: r["cost"]["unknowns"])
    return recs, bad


def main_plot(phase):
    recs, bad = load(phase)
    cases = sorted({c for c, _ in recs})
    mets = METRICS[phase]
    (HERE / "figures").mkdir(exist_ok=True)
    fig, axs = plt.subplots(len(mets) * 2, len(cases), figsize=(3.6 * len(cases), 3.2 * 2 * len(mets)),
                            constrained_layout=True, squeeze=False)  # fmt: skip
    lines = [f"# {phase}: cost to reach targets (log-log interpolation; uncontended n2-standard-2)", ""]
    if bad:
        lines += ["Failed runs: " + "; ".join(f"{n}: {e}" for n, e in bad), ""]
    for j, case in enumerate(cases):
        for i, (met, targets) in enumerate(mets):
            for k, xkey in enumerate(("unknowns", "work")):
                ax = axs[2 * i + k, j]
                for arm, st in STYLE.items():
                    rr = [r for r in recs.get((case, arm), []) if met in r["metrics"]]
                    if not rr:
                        continue
                    x = [r["cost"]["unknowns"] * (r["cost"]["steps"] if xkey == "work" else 1) for r in rr]
                    y = [abs(r["metrics"][met]) for r in rr]
                    ax.loglog(x, y, label=f"Wakis FIT ({arm})", ms=4, lw=1, **st)
                for t in targets:
                    ax.axhline(t, color="0.6", ls=":", lw=0.8)
                ax.set_title(f"{case}: {met}", fontsize=8)
                ax.set_xlabel("unknowns" if xkey == "unknowns" else "work (unknowns × steps)", fontsize=8)
                ax.grid(True, which="both", alpha=0.3)
                ax.tick_params(labelsize=7)
        axs[0, j].legend(fontsize=5.5)
        for met, targets in mets:
            lines += [f"## {case} — {met}", "",
                      "| arm | target | unknowns | work | run_s | best error (unknowns) |",
                      "|---|---|---|---|---|---|"]  # fmt: skip
            for arm in STYLE:
                rr = [r for r in recs.get((case, arm), []) if met in r["metrics"]]
                if not rr:
                    continue
                for t in targets:
                    tr = [dict(r, metrics={met: abs(r["metrics"][met])}) for r in rr]
                    c = runner.cost_to_target(tr, met, t)
                    best = min(rr, key=lambda r: abs(r["metrics"][met]))
                    lines.append(
                        f"| {arm} | {t:g} | {c['unknowns']:,.0f} | {c['work']:.3g} | {c['run_s']:,.0f} "
                        f"| {abs(best['metrics'][met]):.3g} ({best['cost']['unknowns']:,}) |"
                    )
            lines.append("")
    fig.suptitle(f"{phase}: error vs unknowns and vs work (markers: Wakis FIT runs; dotted: targets)")
    fig.savefig(HERE / "figures" / f"{phase}_error.png", dpi=130)
    txt = "\n".join(lines)
    (HERE / "data" / f"{phase}_summary.md").write_text(txt + "\n")
    print(txt)
