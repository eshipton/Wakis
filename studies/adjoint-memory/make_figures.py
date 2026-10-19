"""Figures of the adjoint memory-ladder study (reads the JSON results here)."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).parent
COLORS = {"store": "#444444", "checkpoint": "#1f77b4", "auto": "#17becf",
          "reverse": "#d62728", "nyquist": "#ff7f0e"}  # fmt: skip


def fig_bend(path, out):
    data = json.loads(Path(path).read_text())
    res, meta = data["results"], data["meta"]
    ref = next(r for r in res if r["mode"] == "store" and "store_dtype" not in r["settings"])
    t_ref = ref["t_forward"] + ref["t_backward"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    key = "gpu_peak" if meta["gpu"] else "host_peak"
    for r in res:
        mode = r["mode"]
        mem_gb = r["rung_peak_bytes"] / 1e9
        meas_gb = r[key] / 1e9
        t = (r["t_forward"] + r["t_backward"]) / t_ref
        st = r["settings"]
        short = {"checkpoint": "ck", "auto": "auto", "store": "store",
                 "reverse": "reverse", "nyquist": "nyq"}[mode]  # fmt: skip
        parts = []
        if "checkpoints" in st:
            parts.append(f"s={st['checkpoints']}")
        if "block" in st:
            parts.append(f"b={st['block']}")
        if "budget_gb" in st:
            parts.append(f"{st['budget_gb']} GB")
        if "decimation" in st:
            parts.append(f"d={st['decimation']}")
        if "store_dtype" in st:
            parts.append("f32")
        if "on_host" in st:
            parts.append("host rows")
        label = short + (" " + ",".join(parts) if parts else "")
        marker = "s" if mode == "nyquist" else "o"
        axes[0].scatter(mem_gb, t, color=COLORS[mode], marker=marker, s=36)
        axes[0].annotate(label, (mem_gb, t), fontsize=6.5, xytext=(3, 3),
                         textcoords="offset points")  # fmt: skip
        axes[1].scatter(mem_gb, meas_gb, color=COLORS[mode], marker=marker, s=36)
    axes[0].set(xscale="log", xlabel="rung memory (predicted peak) [GB]",
                ylabel="forward + gradient wall time / store",
                title=f"2D bend, {meta['n']}$^2$ cells, Nt = {meta['Nt']}, "
                      f"{meta['design_edges']} design edges ({'L4 GPU' if meta['gpu'] else 'CPU'})")  # fmt: skip
    lo = min(r["rung_peak_bytes"] for r in res) / 1e9
    hi = max(r["rung_peak_bytes"] for r in res) / 1e9
    axes[1].plot([lo, hi], [lo, hi], "k--", lw=0.8, label="measured = predicted")
    axes[1].set(xscale="log", yscale="log", xlabel="predicted rung peak [GB]",
                ylabel=f"measured {'GPU pool' if meta['gpu'] else 'tracemalloc'} peak [GB]",
                title="high-water mark (whole forward + vjp)")  # fmt: skip
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)


def fig_roundoff(path, out):
    res = json.loads(Path(path).read_text())
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for bc, m in (("cpml", "o"), ("pec", "s")):
        rows = [r for r in res if r["bc"] == bc]
        Nt = np.array([r["Nt"] for r in rows])
        err = np.array([max(r["rel_err"], 1e-18) for r in rows])
        ax.loglog(Nt, err, m + "-", label={"cpml": "open (CPML, recorded layers)",
                                           "pec": "closed PEC cavity"}[bc])  # fmt: skip
    x = np.array([2e2, 1e5])
    ax.loglog(x, 2.2e-16 * x, "k:", lw=0.8, label=r"$\epsilon\,N_t$")
    ax.loglog(x, 2.2e-16 * np.sqrt(x), "k--", lw=0.8, label=r"$\epsilon\,\sqrt{N_t}$")
    ax.set(xlabel="time steps $N_t$", ylabel="max |g_rev - g_store| / max |g_store|",
           title="time-reversed leapfrog: gradient round-off (float64)")  # fmt: skip
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)


def fig_projection(path, out):
    res = json.loads(Path(path).read_text())
    names = {"ceviche_waveguide_bend": "ceviche bend (r=1)",
             "ceviche_wdm": "ceviche WDM (r=1)", "metalens_3d": "3D metalens-sized"}  # fmt: skip
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=False)
    for ax, (key, title) in zip(axes, names.items()):
        p = res[key]
        lad, Nt = p["ladder"], p["Nt"]
        curve = lad["checkpoint_curve"]
        ax.plot([c["bytes"] / 1e9 for c in curve], [c["extra"] / Nt for c in curve],
                "o-", color=COLORS["checkpoint"], ms=3, label="checkpoint (exact)")  # fmt: skip
        for name, style in (("store_f64", ("store float64", "k", "s")),
                            ("store_f32", ("store float32 (1e-7)", "0.5", "s")),
                            ("reverse", ("reverse (lossless only)", COLORS["reverse"], "D")),
                            ("nyquist", ("nyquist (approx.)", COLORS["nyquist"], "^"))):  # fmt: skip
            label, color, marker = style
            ax.scatter(lad[name]["bytes"] / 1e9, lad[name]["extra"] / Nt, color=color,
                       marker=marker, s=50, label=label, zorder=3)  # fmt: skip
        ax.set(xscale="log", yscale="symlog", xlabel="memory for the gradient [GB]",
               ylabel="forward steps recomputed in the backward sweep / Nt",
               title=f"{title}: {np.prod(p['cells']) / 1e6:.2f} Mcells, Nt={Nt}")  # fmt: skip
        ax.grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=150)


if __name__ == "__main__":
    for name, fn in (("bend_gpu.json", fig_bend), ("bend_cpu.json", fig_bend),
                     ("roundoff.json", fig_roundoff),
                     ("projection.json", fig_projection)):  # fmt: skip
        if (HERE / name).exists():
            fn(HERE / name, HERE / name.replace(".json", ".png"))
            print("wrote", name.replace(".json", ".png"))
