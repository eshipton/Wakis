"""
Publish the O-grid benchmark records (B2, B3) to the shared store and compare
with the baselines there at equal unknowns / work (wakis.benchmarks.results).

Usage::

    python bench_compare.py publish DIR --job JOB    # records from a Batch run
    python bench_compare.py compare                  # tables + figures
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
DATA = HERE / "data"
RENAME = {"ogrid": "nfit-ogrid", "ogrid-r0.5": "nfit-ogrid-r0.5", "ogrid-boxed": "nfit-boxed",
          "ogrid-r0.5-o2": "nfit-ogrid-r0.5-o2"}
TARGETS = {
    "B2": ("q_ext_nrms", (0.01, 0.001)),
    "B3": ("enh_peak", (0.05, 0.02)),
}


def publish(d, job):
    from wakis.benchmarks.results import publish as pub

    n = 0
    for f in sorted(Path(d).glob("B[23]-*.json")):
        rec = json.loads(f.read_text())
        rec["arm"] = RENAME.get(rec["arm"], rec["arm"])
        rec["batch_job"] = job
        if "error" in rec:
            print("ERROR", f.name, rec["error"])
            continue
        pub(rec, machine="n2-standard-2 (spot, 1 task per VM)",
            notes="wakis.curvilinear O-grid NFIT, branch feature/curvilinear-nfit")  # fmt: skip
        n += 1
    print("published", n)


def interp(x, e, t):
    o = np.argsort(x)
    x, e = np.asarray(x)[o], np.asarray(e)[o]
    for i in range(len(x) - 1):
        if (e[i] - t) * (e[i + 1] - t) <= 0 and e[i] != e[i + 1]:
            s = (np.log(t) - np.log(e[i])) / (np.log(e[i + 1]) - np.log(e[i]))
            return float(np.exp(np.log(x[i]) + s * (np.log(x[i + 1]) - np.log(x[i]))))
    return np.nan


def compare():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wakis.benchmarks.results import fetch

    FIG.mkdir(exist_ok=True)
    lines = []
    plots = [("B2-Au-TE", "B2"), ("B2-Au-TM", "B2"), ("B2-Ag-TE", "B2"), ("B2-Ag-TM", "B2"),
             ("B3-g2", "B3"), ("B3-g5", "B3"), ("B3-g10", "B3")]  # fmt: skip
    fig, axs = plt.subplots(2, 4, figsize=(17, 7.5), constrained_layout=True)
    for ax, (case, ph) in zip(axs.ravel(), plots):
        metric, targets = TARGETS[ph]
        df = fetch(case)
        if df is None or len(df) == 0:
            continue
        df = df[df["error"].isna()] if "error" in df else df
        lines += [f"### {case} ({metric})", "",
                  "| arm | unknowns @ " + " | unknowns @ ".join(f"{100 * t:g} %" for t in targets)
                  + " | work @ " + " | work @ ".join(f"{100 * t:g} %" for t in targets)
                  + " | time @ " + " | time @ ".join(f"{100 * t:g} %" for t in targets) + " | best |",
                  "|---" * (2 + 3 * len(targets)) + "|"]  # fmt: skip
        for arm, g in df.groupby("arm"):
            e = g[f"m.{metric}"].abs().to_numpy()
            u = g["c.unknowns"].to_numpy()
            w = g["work"].to_numpy()
            ts = g["c.run_s"].to_numpy()
            ok = np.isfinite(e)
            vals = [interp(u[ok], e[ok], t) for t in targets] + [
                interp(w[ok], e[ok], t) for t in targets] + [interp(ts[ok], e[ok], t) for t in targets]
            fmt = lambda v: "—" if np.isnan(v) else f"{v:.3g}"  # noqa: E731
            lines.append(f"| {arm} | " + " | ".join(fmt(v) for v in vals) + f" | {100 * e[ok].min():.3g} % |")
            o = np.argsort(u)
            mine = arm.startswith("nfit")
            ax.loglog(u[o], 100 * e[o], marker="o" if mine else ".", lw=1.8 if mine else 0.8,
                      label=f"Wakis FIT ({arm})")  # fmt: skip
        for t in targets:
            ax.axhline(100 * t, color="0.6", lw=0.6, ls="--")
        ax.set_title(case, fontsize=9)
        ax.set_xlabel("field unknowns")
        ax.set_ylabel(f"{metric} [%]")
        ax.grid(True, which="both", lw=0.3)
        ax.legend(fontsize=6)
        lines.append("")
    axs.ravel()[-1].axis("off")
    fig.suptitle("Shared benchmark store: error vs unknowns (all arms; nfit-* = this work)", fontsize=10)
    fig.savefig(FIG / "f_benchmarks_B2_B3.png")
    (DATA / "benchmarks_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["publish", "compare"])
    ap.add_argument("dir", nargs="?")
    ap.add_argument("--job")
    a = ap.parse_args()
    sys.path.insert(0, str(HERE.parents[1]))
    if a.mode == "publish":
        publish(a.dir, a.job)
    else:
        compare()
