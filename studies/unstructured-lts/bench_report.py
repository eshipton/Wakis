"""
B2/B3 benchmark records (bench_lts.py): summary table, figure and publishing.

Usage::

    python bench_report.py             # data/bench/summary.md + figures/d_bench.png
    python bench_report.py --publish   # shared store (wakis.benchmarks.results)

Published arm names: ``tri-graded-pml`` (global leapfrog at the minimum dt)
and ``tri-graded-pml-lts2`` (same mesh, lts=2).
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA, FIG = HERE / "data" / "bench", HERE / "figures"
PUB = {"tri-graded": "tri-graded-pml", "tri-graded-lts": "tri-graded-pml-lts2"}
COMMIT = "3f6b667"  # code of the Batch job unstr-lts-bench-20261019-0649


def load():
    recs = {}
    for f in sorted(DATA.glob("*.json")):
        r = json.loads(f.read_text())
        recs[(r["case"], r["arm"], round(float(r["resolution"]) * 1e9, 4))] = r
    return recs


def metric(r):
    m = r.get("metrics", {})
    return m.get("enh_nrms", m.get("q_ext_nrms")) if m else None


def summary(recs):
    cases = sorted({k[0] for k in recs})
    lines = [
        "| case | h [nm] | unknowns | global: steps / err / run [s] "
        "| LTS: steps / err / run [s] | step ratio | time ratio | err metric |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for c in cases:
        for h in sorted({k[2] for k in recs if k[0] == c}, reverse=True):
            g, s = recs.get((c, "tri-graded", h)), recs.get((c, "tri-graded-lts", h))
            if not (g and s) or "error" in g or "error" in s:
                err = (g or {}).get("error") or (s or {}).get("error")
                lines.append(f"| {c} | {h:g} | | {err} | | | | |")
                continue
            name = "enh_nrms" if "enh_nrms" in g["metrics"] else "q_ext_nrms"
            cg, cs = g["cost"], s["cost"]
            lines.append(
                f"| {c} | {h:g} | {cg['unknowns']} "
                f"| {cg['steps']} / {metric(g):.4f} / {cg['run_s']:.0f} "
                f"| {cs['steps']} / {metric(s):.4f} / {cs['run_s']:.0f} "
                f"| {cg['steps'] / cs['steps']:.2f} | {cg['run_s'] / cs['run_s']:.2f} "
                f"| {name} |"
            )
    (DATA / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def figure(recs):
    cases = sorted({k[0] for k in recs})
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for i, c in enumerate(cases):
        col = f"C{i}"
        for arm, ls, mk in (("tri-graded", "-", "o"), ("tri-graded-lts", "--", "s")):
            ks = sorted((k for k in recs if k[0] == c and k[1] == arm
                         and "metrics" in recs[k]), key=lambda k: -k[2])  # fmt: skip
            if not ks:
                continue
            u = [recs[k]["cost"]["unknowns"] * recs[k]["cost"]["steps"] for k in ks]
            t = [recs[k]["cost"]["run_s"] for k in ks]
            e = [metric(recs[k]) for k in ks]
            lab = f"{c} {'LTS' if 'lts' in arm else 'global'}"
            ax[0].loglog(u, e, ls, marker=mk, color=col, ms=4, lw=1, label=lab)
            ax[1].loglog(t, e, ls, marker=mk, color=col, ms=4, lw=1, label=lab)
    ax[0].set_xlabel("unknowns × steps (LTS: fine-region extra work not included)")
    ax[0].set_ylabel("error (B2: Q_ext NRMS; B3: gap |E|² NRMS)")
    ax[0].set_title("(a) error vs work, global (solid) vs LTS p=2 (dashed)", fontsize=9)
    ax[1].set_xlabel("uncontended run time [s] (n2-standard-2, 1 core)")
    ax[1].set_title("(b) error vs wall time", fontsize=9)
    ax[0].legend(fontsize=6, ncol=2)
    fig.tight_layout()
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / "d_bench.png", dpi=160)


def publish(recs):
    from wakis.benchmarks.results import publish as pub

    for r in recs.values():
        if "error" in r:
            continue
        rec = dict(r, arm=PUB[r["arm"]], commit=COMMIT)
        print(pub(rec, machine="n2-standard-2",
                  notes="feature/unstructured-lts; Batch unstr-lts-bench-20261019-0649"))  # fmt: skip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--publish", action="store_true")
    a = ap.parse_args()
    recs = load()
    summary(recs)
    figure(recs)
    if a.publish:
        publish(recs)


if __name__ == "__main__":
    main()
