"""
Figures of the unstructured-GPU study from out/bench_*.json.

  python studies/unstructured-gpu/plot_figures.py out/bench_l4_3d.json out/bench_l4_2d.json [...]

Writes docs/img/unstructured_gpu_throughput.png (ns per unknown-step and
speedup vs problem size) and docs/img/unstructured_gpu_breakdown.png
(per-kernel device time per step, largest 3D and 2D sizes).
"""

import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
IMG = ROOT / "docs" / "img"


def load(paths):
    runs = []
    for p in paths:
        for r in json.loads(pathlib.Path(p).read_text()):
            r["_file"] = pathlib.Path(p).stem
            r["_variant"] = "cusparse" if "cusparse" in r["_file"] else "fused"
            runs.append(r)
    return runs


def _short(g):
    return g.replace("NVIDIA ", "").replace("-SXM4-40GB", "")


def _series(runs, dim, gpu, variant):
    rs = [r for r in runs if r["dim"] == dim and r["gpu_name"] == gpu and r["_variant"] == variant]
    return sorted(rs, key=lambda r: r["n_unknowns"])


def throughput(runs, out):
    gpus = sorted({r["gpu_name"] for r in runs})
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.9), constrained_layout=True)
    for dim, ax in zip((3, 2), axs[:2]):
        for gi, g in enumerate(gpus):
            col = f"C{gi}"
            rs = _series(runs, dim, g, "fused")
            if rs:
                ax.loglog([r["n_unknowns"] for r in rs], [r["gpu_ns_per_unknown_step"] for r in rs],
                          "o-", color=col, label=f"DeviceStepper, {_short(g)}")  # fmt: skip
            rv = _series(runs, dim, g, "cusparse")
            if rv:
                ax.loglog([r["n_unknowns"] for r in rv], [r["gpu_ns_per_unknown_step"] for r in rv],
                          "o:", color=col, mfc="white", alpha=0.6, label=f"v1 (cuSPARSE), {_short(g)}")  # fmt: skip
            if dim == 3 and rs and all("cart" in r for r in rs):
                for c_, mk, ls in ((0, "v", "--"), (1, "^", "-.")):
                    ax.loglog([r["cart"][c_]["n_unknowns"] for r in rs],
                              [r["cart"][c_]["ns_per_unknown_step"] for r in rs], mk + ls, color=col, ms=3.5,
                              label=f"SolverFIT3D {'fused' if c_ else 'CuPy'}, {_short(g)}")  # fmt: skip
        rs = _series(runs, dim, gpus[0], "fused")
        cpu = [(r["n_unknowns"], r["cpu_ns_per_unknown_step"]) for r in rs if "cpu_ns_per_unknown_step" in r]
        if cpu:
            cn, cv = np.array(cpu).T
            ax.loglog(cn, cv, "s-", color="0.35", ms=3.5, label="SciPy 1 core (host)")
        ax.set_xlabel("unknowns")
        ax.set_ylabel("ns per unknown and step")
        ax.set_title(f"({'a' if dim == 3 else 'b'}) {dim}D {'tetrahedra' if dim == 3 else 'triangles, TM'}")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=4.6, loc="upper right")
    bx = axs[2]
    for gi, g in enumerate(gpus):
        col = f"C{gi}"
        for dim, mk in ((3, "o"), (2, "s")):
            rs = [r for r in _series(runs, dim, g, "fused") if "cpu_s_per_step" in r]
            if rs:
                bx.loglog([r["n_unknowns"] for r in rs], [r["cpu_s_per_step"] / r["gpu_s_per_step"] for r in rs],
                          mk + "-", color=col, ms=3.5, label=f"vs SciPy 1 core, {dim}D, {_short(g)}")  # fmt: skip
        rs = _series(runs, 3, g, "fused")
        if rs and all("cart" in r for r in rs):
            for c_, mk in ((0, "v--"), (1, "^-.")):
                bx.loglog([r["n_unknowns"] for r in rs],
                          [r["cart"][c_]["ns_per_unknown_step"] / r["gpu_ns_per_unknown_step"] for r in rs],
                          mk, color=col, ms=3.5,
                          label=f"vs SolverFIT3D {'fused' if c_ else 'CuPy'}, {_short(g)}")  # fmt: skip
    bx.axhline(1, color="0.4", lw=0.8)
    bx.set_xlabel("unknowns")
    bx.set_ylabel("baseline time / DeviceStepper time")
    bx.set_title("(c) speed ratio")
    bx.grid(True, which="both", alpha=0.3)
    bx.legend(fontsize=4.6, loc="upper left")
    fig.savefig(out, dpi=200)
    plt.close(fig)


def breakdown(runs, out):
    """Share of each operation in the device time of one step (CUDA events)."""
    pick = []
    for g in sorted({r["gpu_name"] for r in runs}):
        for dim in (3, 2):
            rs = _series(runs, dim, g, "fused")
            if rs:
                for r in (rs[0], rs[-1]):
                    pick.append((f"{_short(g)} {dim}D\n{r['n_unknowns']:.1e} unk.", r))
    if not pick:
        return
    names = list(pick[0][1]["breakdown_mon_us"])
    fig, ax = plt.subplots(figsize=(7.0, 3.4), constrained_layout=True)
    x = np.arange(len(pick))
    bottom = np.zeros(len(pick))
    tot = np.array([sum(p[1]["breakdown_mon_us"].values()) for p in pick])
    for i, nm in enumerate(names):
        v = np.array([p[1]["breakdown_mon_us"][nm] for p in pick]) / tot * 100
        ax.bar(x, v, bottom=bottom, color=f"C{i}", label=nm, width=0.6, edgecolor="white", lw=0.5)
        bottom += v
    for xi, p, t in zip(x, pick, tot):
        wall = p[1]["gpu_mon_s_per_step"] * 1e6
        ax.text(xi, 102, f"{t:.0f} us\n({wall:.0f} us wall)", ha="center", va="bottom", fontsize=6)
    ax.set_xticks(x, [p[0] for p in pick], fontsize=6.5)
    ax.set_ylim(0, 125)
    ax.set_ylabel("share of device time per step [%]")
    ax.set_title("Per-launch breakdown of the fused step, 1 probe + 3-frequency DFT (CUDA events)", fontsize=8)
    ax.legend(fontsize=6, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    ax.grid(True, axis="y", alpha=0.3)
    fig.savefig(out, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    runs = load(sys.argv[1:])
    IMG.mkdir(parents=True, exist_ok=True)
    throughput(runs, IMG / "unstructured_gpu_throughput.png")
    breakdown(runs, IMG / "unstructured_gpu_breakdown.png")
    print("wrote", IMG / "unstructured_gpu_throughput.png", IMG / "unstructured_gpu_breakdown.png")
