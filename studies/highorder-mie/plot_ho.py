"""Figures and tables for the high-order study (see ho_study.py)."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
BASE = HERE / "baseline"  # copied from feature/tri-fixes 75f603d
BASE_ARMS = ["tri-graded-pml", "tri-graded-compact", "cart-graded-compact"]
TARGETS = [0.02, 0.01, 0.001]


def rms(d):
    q, qr = np.array(d["q"]), np.array(d["q_ref"])
    return float(np.sqrt(np.mean((q / qr - 1) ** 2)))


def load(folder, arms=None):
    runs = {}
    for f in sorted(Path(folder).glob("*.json")):
        d = json.loads(f.read_text())
        if "error" in d or "q" not in d:
            continue
        if arms and d["arm"] not in arms:
            continue
        d["rms"] = rms(d)
        d["work"] = d["dofs"] * d["steps"]
        d["rhs_work"] = d["dofs"] * d.get("rhs_evals", d["steps"])
        runs.setdefault((d["arm"], d["pol"]), []).append(d)
    for k in runs:
        runs[k].sort(key=lambda d: d["dofs"])
    return runs


def at_target(runs, key, target):
    """Log-log interpolation of `key` at RMS = target; extrapolated from the
    last two points (flag True) when the target is not bracketed."""
    xs = np.array([d[key] for d in runs], float)
    es = np.array([d["rms"] for d in runs])
    for i in range(len(es) - 1):
        if (es[i] - target) * (es[i + 1] - target) <= 0 and es[i] != es[i + 1]:
            t = np.log(target / es[i]) / np.log(es[i + 1] / es[i])
            return float(np.exp(np.log(xs[i]) + t * np.log(xs[i + 1] / xs[i]))), False
    if len(es) >= 2 and es[-1] > target and es[-1] < es[-2]:
        slope = np.log(es[-1] / es[-2]) / np.log(xs[-1] / xs[-2])
        return float(xs[-1] * (target / es[-1]) ** (1 / slope)), True
    if es.min() <= target:
        return float(xs[np.argmax(es <= target)]), False
    return np.nan, True


def label(arm):
    if arm.startswith("dg"):
        p = arm[2]
        tail = arm[3:]
        mesh = {"": "curved, PML box", "-compact": "curved", "-straight": "straight-sided"}.get(tail, tail)
        return f"Wakis DGTD p={p} ({mesh})"
    return {
        "fd22": "Wakis FD (2,2) Yee",
        "fd24": "Wakis FD (2,4) leapfrog",
        "fd24rk": "Wakis FD (2,4) LSRK4",
        "fd24-c30": "Wakis FD (2,4) leapfrog, CFL 0.3",
        "tri-graded-pml": "Wakis FIT (graded triangles, PML)",
        "tri-graded-compact": "Wakis FIT (graded triangles, compact)",
        "cart-graded-compact": "Wakis FIT (graded Cartesian, conformal)",
    }.get(arm, arm)


def table(runs, base, targets=TARGETS):
    lines = ["| pol | arm | target | unknowns | unknowns × steps | unknowns × RHS evals | time [s] |",
             "|---|---|---|---|---|---|---|"]  # fmt: skip
    allr = dict(base)
    allr.update(runs)
    for pol in ("TM", "TE"):
        for (arm, p), rs in sorted(allr.items()):
            if p != pol or len(rs) < 2:
                continue
            for tg in targets:
                cells = []
                for key in ("dofs", "work", "rhs_work", "run_s"):
                    v, ex = at_target(rs, key, tg)
                    cells.append("–" if not np.isfinite(v) else (f"{v:.3g}" + ("*" if ex else "")))
                lines.append(f"| {pol} | {arm} | {tg * 100:g} % | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def runs_table(runs):
    lines = ["| arm | pol | res | unknowns | cells | dt [as] | steps | run [s] | ns/(unknown·step) | RMS err | max err |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for (arm, pol), rs in sorted(runs.items()):
        for d in rs:
            q, qr = np.array(d["q"]), np.array(d["q_ref"])
            lines.append(
                f"| {arm} | {pol} | {d['res']:g} | {d['dofs']:,} | {d['cells']:,} | "
                f"{d['dt'] * 1e18:.1f} | {d['steps']} | {d['run_s']:.0f} | "
                f"{d['run_s'] / d['work'] * 1e9:.1f} | {d['rms'] * 100:.3f} % | "
                f"{np.abs(q / qr - 1).max() * 100:.2f} % |"
            )
    return "\n".join(lines)


STYLE = {
    "dg2-compact": ("C0", "v"), "dg3-compact": ("C1", "o"), "dg4-compact": ("C2", "s"),
    "dg5-compact": ("C3", "D"), "dg3": ("C1", "x"), "dg4": ("C2", "+"),
    "dg4-straight": ("C7", "s"), "fd22": ("C4", "^"), "fd24": ("C5", ">"),
    "fd24rk": ("C6", "<"), "tri-graded-pml": ("k", "o"), "tri-graded-compact": ("0.4", "o"),
    "cart-graded-compact": ("0.6", "s"),
}  # fmt: skip


def fig_error(runs, base, out, title, keys=("dofs", "rhs_work", "run_s")):
    xl = {"dofs": "field unknowns", "work": "unknowns × steps",
          "rhs_work": "unknowns × RHS evaluations", "run_s": "1-core run time [s]"}  # fmt: skip
    fig, axs = plt.subplots(2, len(keys), figsize=(5 * len(keys), 8), squeeze=False)
    allr = dict(base)
    allr.update(runs)
    for i, pol in enumerate(("TM", "TE")):
        for j, key in enumerate(keys):
            ax = axs[i, j]
            for (arm, p), rs in sorted(allr.items()):
                if p != pol or arm not in STYLE:
                    continue
                c, m = STYLE[arm]
                x = [d[key] for d in rs]
                ax.loglog(x, [d["rms"] * 100 for d in rs], marker=m, color=c,
                          ls="--" if arm in BASE_ARMS else "-", label=label(arm))  # fmt: skip
            for tg in TARGETS:
                ax.axhline(tg * 100, color="0.8", lw=0.8, zorder=0)
            ax.set_xlabel(xl[key])
            ax.set_ylabel(f"RMS error of Q_sca, {pol} [%]")
            ax.grid(True, which="both", alpha=0.2)
    axs[0, 0].legend(fontsize=7, loc="lower left")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def fig_qsca(runs, out, arm, res_pick, title, xs=None):
    import sys

    sys.path.insert(0, str(HERE.parent / "mie-cylinder-mesh"))
    fig, axs = plt.subplots(2, 2, figsize=(11, 7), sharex=True,
                            gridspec_kw=dict(height_ratios=[3, 1.3]))  # fmt: skip
    for j, pol in enumerate(("TM", "TE")):
        rs = [d for d in runs.get((arm, pol), []) if d["res"] == res_pick]
        if not rs:
            continue
        d = rs[0]
        q, qr = np.array(d["q"]), np.array(d["q_ref"])
        x = xs if xs is not None else np.linspace(2, 10, len(q))
        axs[0, j].plot(x, qr, "k-", lw=1.2, label="exact series")
        axs[0, j].plot(x, q, "o", ms=4, color="C2", label=label(arm) + f", res {res_pick:g}")
        axs[1, j].semilogy(x, np.abs(q / qr - 1) * 100, "o-", ms=3, color="C2")
        axs[0, j].set_title(f"{pol}: RMS {d['rms'] * 100:.3f} %, {d['dofs']:,} unknowns")
        axs[0, j].set_ylabel("Q_sca")
        axs[1, j].set_ylabel("|error| [%]")
        axs[1, j].set_xlabel("size parameter x = k a")
        axs[0, j].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main_plot():
    FIG.mkdir(exist_ok=True)
    base = load(BASE, BASE_ARMS)
    runs = load(HERE / "data")
    if runs:
        fig_error(runs, base, FIG / "a_error_vs_cost.png",
                  "Mie cylinder n = 2, x = 2-10: high-order vs best second-order")  # fmt: skip
        best = "dg4-compact"
        res = sorted({d["res"] for d in runs.get((best, "TM"), [])})
        if res:
            fig_qsca(runs, FIG / "b_qsca_dg4.png", best, res[min(1, len(res) - 1)],
                     "Q_sca vs x: DGTD p = 4 on curved triangles")  # fmt: skip
        txt = ["# Data summary (generated by plot_ho.py)", "", "## Runs", "",
               runs_table(runs), "", "## Cost to reach a target RMS error", "",
               "(* = extrapolated from the last two runs; baselines from ../mie-cylinder-graded)",
               "", table(runs, base)]  # fmt: skip
        (HERE / "data" / "summary.md").write_text("\n".join(txt) + "\n")
        error_split(runs, FIG / "f_error_split.png", HERE / "data" / "error_split.md")
    large = load(HERE / "data_large")
    if large:
        fig_error(large, {}, FIG / "c_large_error_vs_cost.png",
                  "Lossy n = 2 cylinder, x = 20-30 (showcase)")  # fmt: skip
        fig_qsca(large, FIG / "d_large_qsca.png", "dg4-compact",
                 sorted({d["res"] for d in large.get(("dg4-compact", "TM"), [])})[-1],
                 "Showcase x = 20-30", xs=np.linspace(20, 30, 41))  # fmt: skip
        txt = ["# Showcase data (generated by plot_ho.py)", "", runs_table(large), "",
               table(large, {}, [0.01, 0.001])]  # fmt: skip
        (HERE / "data_large" / "summary.md").write_text("\n".join(txt) + "\n")


if __name__ == "__main__":
    main_plot()


def error_split(runs, out_png, out_md):
    """Bulk-dispersion vs interface error of the Cartesian schemes.

    fd22 and fd24rk share the grid, the sub-pixel averaging, the source and
    the extraction; only the bulk stencil (and the time integrator) differ.
    With signed per-frequency errors e = Q/Q_ref - 1, e(fd24rk) estimates the
    interface (+ residual 4th-order bulk) error and e(fd22) - e(fd24rk) the
    second-order bulk dispersion of Yee. RMS over x <= 6 and x > 6.
    """
    x = np.linspace(2, 10, 41)
    lo, hi = x <= 6, x > 6
    rows = ["| pol | res | RMS Yee total | RMS bulk (Yee - (2,4)RK) | RMS interface ((2,4)RK) | "
            "bulk share x>6 | interface x>6 | bulk x<=6 | interface x<=6 |",
            "|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for j, pol in enumerate(("TM", "TE")):
        a = {d["res"]: d for d in runs.get(("fd22", pol), [])}
        b = {d["res"]: d for d in runs.get(("fd24rk", pol), [])}
        rr, bulk, intf = [], [], []
        for res in sorted(set(a) & set(b)):
            e2 = np.array(a[res]["q"]) / np.array(a[res]["q_ref"]) - 1
            e4 = np.array(b[res]["q"]) / np.array(b[res]["q_ref"]) - 1
            eb = e2 - e4

            def r(v, m=slice(None)):
                return np.sqrt(np.mean(v[m] ** 2)) * 100

            rows.append(f"| {pol} | {res:g} | {r(e2):.3f} % | {r(eb):.3f} % | {r(e4):.3f} % | "
                        f"{r(eb, hi):.3f} % | {r(e4, hi):.3f} % | {r(eb, lo):.3f} % | {r(e4, lo):.3f} % |")  # fmt: skip
            rr.append(res)
            bulk.append(r(eb, hi))
            intf.append(r(e4, hi))
        if rr:
            axs[j].loglog(rr, bulk, "o-", label="bulk dispersion of Yee (2,2), x > 6")
            axs[j].loglog(rr, intf, "s-", label="interface + residual, (2,4) LSRK4, x > 6")
            rr = np.array(rr, float)
            axs[j].loglog(rr, bulk[0] * (rr / rr[0]) ** -2, "k:", lw=0.8, label="order 2")
            axs[j].loglog(rr, intf[0] * (rr / rr[0]) ** -2, "k:", lw=0.8)
        axs[j].set_xlabel("cells per dielectric wavelength (x = 10)")
        axs[j].set_ylabel("RMS error of Q_sca [%]")
        axs[j].set_title(f"{pol}: Wakis FD error split (same grid, averaging, source)")
        axs[j].legend(fontsize=7)
        axs[j].grid(True, which="both", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    Path(out_md).write_text("\n".join(rows) + "\n")
