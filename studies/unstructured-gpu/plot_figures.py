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
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
IMG = ROOT / "docs" / "img"


def load(paths):
    runs = []
    for p in paths:
        for r in json.loads(pathlib.Path(p).read_text()):
            r["_file"] = pathlib.Path(p).stem
            runs.append(r)
    return runs


def throughput(runs, out):
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.0, 3.1), constrained_layout=True)
    groups = {}
    for r in runs:
        groups.setdefault((r["dim"], r["gpu_name"]), []).append(r)
    mk = {3: "o", 2: "s"}
    for gi, ((dim, gname), rs) in enumerate(sorted(groups.items())):
        rs = sorted(rs, key=lambda r: r["n_unknowns"])
        n = np.array([r["n_unknowns"] for r in rs])
        g = np.array([r["gpu_ns_per_unknown_step"] for r in rs])
        tag = f"{dim}D {'tets' if dim == 3 else 'TM tris'}, {gname.replace('NVIDIA ', '')}"
        ax.loglog(n, g, mk[dim] + "-", color=f"C{gi}", label=f"DeviceStepper, {tag}")
        cpu = [(r["n_unknowns"], r["cpu_ns_per_unknown_step"]) for r in rs if "cpu_ns_per_unknown_step" in r]
        if cpu:
            cn, cv = np.array(cpu).T
            ax.loglog(cn, cv, mk[dim] + "--", color=f"C{gi}", mfc="white",
                      label=f"SciPy 1 core, {dim}D")  # fmt: skip
            sp = {r["n_unknowns"]: r["cpu_s_per_step"] / r["gpu_s_per_step"] for r in rs if "cpu_s_per_step" in r}
            bx.semilogx(list(sp), list(sp.values()), mk[dim] + "-", color=f"C{gi}",
                        label=f"vs SciPy 1 core, {tag}")  # fmt: skip
        if dim == 3 and all("cart" in r for r in rs):
            for c_ in (0, 1):
                cn = [r["cart"][c_]["n_unknowns"] for r in rs]
                cv = [r["cart"][c_]["ns_per_unknown_step"] for r in rs]
                lab = "SolverFIT3D GPU " + ("fused" if c_ else "CuPy")
                ax.loglog(cn, cv, "^:" if c_ else "v:", color=f"C{gi + 3 + c_}", label=f"{lab}, {gname.replace('NVIDIA ', '')}")
                ratio = [r["gpu_ns_per_unknown_step"] / r["cart"][c_]["ns_per_unknown_step"] for r in rs]
                bx.semilogx(n, 1 / np.array(ratio), "^:" if c_ else "v:", color=f"C{gi + 3 + c_}",
                            label=f"vs {lab} (equal unknowns)")  # fmt: skip
    ax.set_xlabel("unknowns (state entries)")
    ax.set_ylabel("time per unknown and step [ns]")
    ax.set_title("(a) cost per unknown-step")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=5.5, loc="upper right")
    bx.axhline(1, color="0.4", lw=0.8)
    bx.set_yscale("log")
    bx.set_xlabel("unknowns (state entries)")
    bx.set_ylabel("speed ratio (baseline time / DeviceStepper time)")
    bx.set_title("(b) DeviceStepper speed vs baselines")
    bx.grid(True, which="both", alpha=0.3)
    bx.legend(fontsize=5.5, loc="best")
    fig.savefig(out, dpi=200)
    plt.close(fig)


def breakdown(runs, out):
    pick = []
    for dim in (3, 2):
        rs = [r for r in runs if r["dim"] == dim]
        if rs:
            gn = sorted({r["gpu_name"] for r in rs})[0]
            rs = [r for r in rs if r["gpu_name"] == gn]
            small = min(rs, key=lambda r: r["n_unknowns"])
            big = max(rs, key=lambda r: r["n_unknowns"])
            pick += [(f"{dim}D small\n{small['n_unknowns']:.1e} unk.", small),
                     (f"{dim}D large\n{big['n_unknowns']:.1e} unk.", big)]  # fmt: skip
    if not pick:
        return
    names = list(pick[0][1]["breakdown_mon_us"])
    fig, ax = plt.subplots(figsize=(7.0, 3.2), constrained_layout=True)
    x = np.arange(len(pick))
    bottom = np.zeros(len(pick))
    for i, nm in enumerate(names):
        v = np.array([p[1]["breakdown_mon_us"][nm] / p[1]["n_unknowns"] * 1e3 for p in pick])
        ax.bar(x, v, bottom=bottom, color=f"C{i}", label=nm, width=0.6)
        bottom += v
    tot = np.array([p[1]["gpu_mon_ns_per_unknown_step"] for p in pick])
    ax.plot(x, tot, "k_", ms=30, mew=2, label="wall time per step (unsynchronised loop)")
    ax.set_xticks(x, [p[0] for p in pick])
    ax.set_ylabel("device time per unknown and step [ns]")
    ax.set_title(f"Per-kernel breakdown (CUDA events; with 1 probe + 3-frequency DFT), {pick[0][1]['gpu_name']}")
    ax.legend(fontsize=6, ncol=2, loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.savefig(out, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    runs = load(sys.argv[1:])
    IMG.mkdir(parents=True, exist_ok=True)
    throughput(runs, IMG / "unstructured_gpu_throughput.png")
    breakdown(runs, IMG / "unstructured_gpu_breakdown.png")
    print("wrote", IMG / "unstructured_gpu_throughput.png", IMG / "unstructured_gpu_breakdown.png")
