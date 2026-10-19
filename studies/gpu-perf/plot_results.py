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
    ax.plot(x, cpu, ls="none", marker="_", color="k", ms=22, mew=2,
            label="CPU, 1 core (develop)")  # fmt: skip
    ax.set_yscale("log")
    ax.set_ylabel("time per step [ms]")
    ax.set_title("Wakis GPU time step on an NVIDIA L4: develop vs fused kernels "
                 "(lower is better)")  # fmt: skip
    ax.legend(fontsize=8, ncol=1, loc="upper left", framealpha=1)
    ax.grid(axis="y", which="both", alpha=0.3)
    ax = axes[1]
    for k, (tag, label, color) in enumerate(TAGS[:2]):
        n = [launches(tag, c) for c, _ in CASES]
        ax.bar(x + (k - 1) * w, n, w, color=color)
    ax.text(0.99, 0.92, "with CUDA graphs: one graph launch per 64 steps",
            transform=ax.transAxes, ha="right", fontsize=8, color=TAGS[2][2])  # fmt: skip
    ax.set_yscale("log")
    ax.set_ylabel("kernel launches\nper step (host API)")
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in CASES], fontsize=8)
    ax.grid(axis="y", which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(HERE / "gpu_perf_step_time.png", dpi=150)


def fig_breakdown():
    comps = ["sources", "step", "monitors"]
    cases = [c for c, _ in CASES if c not in ("adjoint2d", "bend")]
    labels = [lab for c, lab in CASES if c in cases]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    y = np.arange(len(cases))
    h = 0.38
    for ax, comp in zip(axes, comps):
        for k, (tag, label, color) in enumerate(TAGS[:2]):
            v = []
            for c in cases:
                r = _load(tag, c)
                v.append(np.nan if r is None else r["breakdown_ms"][comp])
            ax.barh(y + (k - 0.5) * h, v, h, color=color, label=label)
            for yi, vi in zip(y, v):
                ax.text(vi * 1.1, yi + (k - 0.5) * h, f"{vi:.2g}", va="center",
                        fontsize=7)  # fmt: skip
        ax.set_xscale("log")
        ax.set_xlim(1e-3, 30)
        ax.set_title(comp)
        ax.grid(axis="x", which="both", alpha=0.3)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([lab.replace("\n", " ") for lab in labels], fontsize=8)
    axes[0].invert_yaxis()
    axes[2].legend(fontsize=8, loc="center right")
    fig.supxlabel("ms per step (device synchronized after each component; empty "
                  "monitor lists cost only the Python loop)", fontsize=9)
    fig.suptitle("Per-component cost of one time step, develop (CuPy) vs fused")
    fig.tight_layout()
    fig.savefig(HERE / "gpu_perf_breakdown.png", dpi=150)


if __name__ == "__main__":
    fig_step_time()
    fig_breakdown()
