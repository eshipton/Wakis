"""Figures of the GPU time-stepping study from results/*.json and prof/*.csv.

  python plot_results.py     # writes gpu_perf_step_time.png, gpu_perf_breakdown.png

Tags: base (develop, CuPy path), fused (fused=True, per-step Python loop),
graph (fused=True through emsolve / AdjointSimulation, CUDA graphs).
"""

import csv
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
RES, PROF = HERE / "results", HERE / "prof"

CASES = [
    ("plain3d", "(a) plain 3D\n275 kcells"),
    ("plain3d_big", "(a') plain 3D\n2.1 Mcells"),
    ("cpml3d", "(b) CPML 3D\n275 kcells"),
    ("tensor2d", "(c) tensor 2D\n66 kcells"),
    ("ade3d", "(c') ADE 3D\n275 kcells"),
    ("tfsf2d", "(d) TF/SF + 81-f\nflux 2D, 44 kcells"),
    ("adjoint2d", "(e) adjoint\n2D 26 kcells"),
    ("bend", "(f) invrs bend\n10 nm, fwd+adj"),
]
TAGS = [("base", "develop (CuPy)", "#9e9e9e"),
        ("fused", "fused, Python loop", "#4c9be8"),
        ("graph", "fused + CUDA graphs", "#1b5e9e")]  # fmt: skip


def _load(tag, case):
    name = f"{tag}_{case}.json" if case == "bend" else f"{tag}_{case}_gpu.json"
    p = RES / name
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _ms(r, case):
    if r is None:
        return np.nan
    if case == "adjoint2d":
        return 0.5 * (r["forward_ms_per_step"] + r["adjoint_ms_per_step"])
    return r["ms_per_step"]


def _cpu(case):
    p = RES / f"base_{case}_cpu.json"
    if not p.exists():
        return np.nan
    return _ms(json.loads(p.read_text()), case)


def launches(tag, case):
    p = PROF / f"{tag}_{case}_cuda_api_sum.csv"
    if not p.exists():
        return np.nan
    rows = list(csv.reader(p.open()))
    n = sum(int(r[2]) for r in rows[1:] if r and "Launch" in r[-1])
    steps = 4000 if case == "adjoint2d" else 100
    return n / steps


def fig_step_time():
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})  # fmt: skip
    x = np.arange(len(CASES))
    w = 0.26
    ax = axes[0]
    for k, (tag, label, color) in enumerate(TAGS):
        ms = [_ms(_load(tag, c), c) for c, _ in CASES]
        bars = ax.bar(x + (k - 1) * w, ms, w, label=label, color=color)
        for b, v in zip(bars, ms):
            if np.isfinite(v):
                ax.text(b.get_x() + b.get_width() / 2, v * 1.08, f"{v:.2g}",
                        ha="center", va="bottom", fontsize=7)  # fmt: skip
    cpu = [_cpu(c) for c, _ in CASES]
    ax.plot(x, cpu, "k_", ms=22, mew=2, label="CPU, 1 core (develop)")
    ax.set_yscale("log")
    ax.set_ylabel("time per step [ms]")
    ax.set_title("Wakis GPU time step on an NVIDIA L4: develop vs fused kernels "
                 "(lower is better)")  # fmt: skip
    ax.legend(fontsize=8, ncol=2, loc="upper left")
    ax.grid(axis="y", which="both", alpha=0.3)
    ax = axes[1]
    for k, (tag, label, color) in enumerate(TAGS):
        n = [launches(tag, c) for c, _ in CASES]
        ax.bar(x + (k - 1) * w, n, w, color=color)
    ax.set_yscale("log")
    ax.set_ylabel("kernel launches\nper step (host API)")
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in CASES], fontsize=8)
    ax.grid(axis="y", which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(HERE / "gpu_perf_step_time.png", dpi=150)


def fig_breakdown():
    comps = ["sources", "step", "monitors"]
    colors = ["#f4a261", "#2a9d8f", "#8e7dbe"]
    cases = [c for c, _ in CASES if c not in ("adjoint2d", "bend")]
    labels = [lab for c, lab in CASES if c in cases]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(cases))
    w = 0.38
    for k, tag in enumerate(("base", "fused")):
        bottom = np.zeros(len(cases))
        for comp, col in zip(comps, colors):
            v = []
            for c in cases:
                r = _load(tag, c)
                v.append(np.nan if r is None else r["breakdown_ms"][comp])
            v = np.nan_to_num(np.asarray(v))
            ax.bar(x + (k - 0.5) * w, v, w, bottom=bottom, color=col,
                   edgecolor="k" if tag == "fused" else "none",
                   label=comp if k == 0 else None, alpha=0.6 if tag == "base" else 1)  # fmt: skip
            bottom += v
        for xi, b in zip(x, bottom):
            ax.text(xi + (k - 0.5) * w, b * 1.05, "dev" if tag == "base" else "fused",
                    ha="center", fontsize=7)  # fmt: skip
    ax.set_yscale("log")
    ax.set_ylabel("time per step, synchronized\nafter each component [ms]")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.legend(fontsize=8)
    ax.set_title("Per-component cost (left bar: develop, right bar: fused)")
    ax.grid(axis="y", which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(HERE / "gpu_perf_breakdown.png", dpi=150)


if __name__ == "__main__":
    fig_step_time()
    fig_breakdown()
