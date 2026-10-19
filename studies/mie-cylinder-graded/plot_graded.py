"""Figures and tables of the graded-mesh Mie study (reads data/*.json and the
uniform-mesh data of ../mie-cylinder-mesh/data)."""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "mie-cylinder-mesh"))
import graded_study as G  # noqa: E402
import mie_mesh_study as S  # noqa: E402

DATA = HERE / "data"
FIG = HERE / "figures"
UNIFORM = HERE.parent / "mie-cylinder-mesh" / "data"
STYLE = {
    "cart-uniform": dict(color="#E69F00", marker="D", ls="--", label="Wakis FIT (Cartesian uniform, conformal)"),
    "cart-graded": dict(color="#D55E00", marker="s", ls="-", label="Wakis FIT (Cartesian graded, conformal)"),
    "cart-graded-compact": dict(color="#CC79A7", marker="v", ls="-", label="Wakis FIT (Cartesian graded, compact domain)"),
    "tri-uniform": dict(color="#56B4E9", marker="o", ls="--", label="Wakis FIT (triangles uniform, Voronoi)"),
    "tri-graded": dict(color="#0072B2", marker="o", ls="-", label="Wakis FIT (triangles graded, Voronoi)"),
    "tri-graded-compact": dict(color="#009E73", marker="^", ls="-", label="Wakis FIT (triangles graded, compact domain)"),
}  # fmt: skip
TARGETS = (0.02, 0.01)
plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 150, "font.size": 9})


def _metrics(d):
    q, r = np.array(d["q"]), np.array(d["q_ref"])
    d["err_rms"] = float(np.sqrt(np.mean((q / r - 1) ** 2)))
    d["err_max"] = float(np.max(np.abs(q / r - 1)))
    d["refl_max"] = float(np.max(d["refl"]))
    return d


def load():
    """Accuracy runs per (arm, pol, res); timing runs keyed separately."""
    acc, timing = {}, {}
    for f in sorted(DATA.glob("*.json")):
        d = json.loads(f.read_text())
        if "error" in d:
            print("FAILED", f.name, d["error"])
            continue
        _metrics(d)
        key = (d["arm"], d["pol"], d["res"])
        if d["arm"].endswith("uniform"):
            timing[key] = d
        else:
            acc[key] = d
    # uniform accuracy: the first study (identical setup, contended timings)
    for name, arm in (("conformal", "cart-uniform"), ("tri-voronoi", "tri-uniform")):
        for f in UNIFORM.glob(f"{name}_T?_*.json"):
            if "flux" in f.name:
                continue
            d = _metrics(json.loads(f.read_text()))
            d["arm"] = arm
            acc[(arm, d["pol"], d["res"])] = d
    return acc, timing


def series_of(acc, arm, pol):
    rr = sorted(
        (d for (a, p, _), d in acc.items() if a == arm and p == pol),
        key=lambda d: d["res"],
    )
    return rr


def cost_per_update(timing, acc):
    """Seconds per (unknown x step) of each solver family, from uncontended runs."""
    out = {}
    for fam in ("cart", "tri"):
        v = [
            d["run_s"] / (d["dofs"] * d["steps"])
            for (a, _, _), d in timing.items()
            if a.startswith(fam)
        ]
        if not v:  # fall back to the graded accuracy runs (also uncontended)
            v = [d["run_s"] / (d["dofs"] * d["steps"]) for (a, _, _), d in acc.items()
                 if a.startswith(fam) and not a.endswith("uniform")]  # fmt: skip
        out[fam] = (float(np.median(v)), float(np.min(v)), float(np.max(v)), len(v))
    return out


def to_target(rr, key, target):
    """log-log interpolation of `key` where the RMS error crosses `target`."""
    e = np.array([d["err_rms"] for d in rr])
    v = np.array(
        [float(d[key]) if key != "work" else float(d["dofs"]) * d["steps"] for d in rr]
    )
    for i in range(len(rr) - 1):
        if e[i] >= target >= e[i + 1]:
            t = (np.log(target) - np.log(e[i])) / (np.log(e[i + 1]) - np.log(e[i]))
            return float(np.exp(np.log(v[i]) + t * (np.log(v[i + 1]) - np.log(v[i]))))
    return np.nan


# --------------------------------------------------------------------------
def fig_meshes(res=10):
    import wakis  # noqa: F401
    from matplotlib.collections import LineCollection

    from wakis.unstructured import disk_mesh, graded_disk_mesh

    h = S.LAM_MIN_DIEL / res
    fig, axs = plt.subplots(1, 4, figsize=(14, 3.9), constrained_layout=True)
    th = np.linspace(0, np.pi / 2, 100)
    for ax, graded in zip(axs[:2], (False, True)):
        r_out = G.GEOM["std"]["r_out"]
        m = (
            graded_disk_mesh([S.A, r_out], G.size_field(h, True))
            if graded
            else disk_mesh([S.A, r_out], h)
        )
        X = m.nodes * 1e6
        seg = X[m.edges]
        keep = (seg[:, :, 0].max(1) > -0.05) & (seg[:, :, 1].max(1) > -0.05)
        ax.add_collection(LineCollection(seg[keep], colors="#0072B2", linewidths=0.25))
        ax.set_title(
            f"triangles {'graded' if graded else 'uniform'}: {m.n_nodes:,} nodes"
        )
    for ax, graded in zip(axs[2:], (False, True)):
        xn = G.axis_nodes(h, graded, G.GEOM["std"]["r_out"], S.N_PML) * 1e6
        xp = xn[xn >= -1e-9]
        for v in xp:
            ax.plot([v, v], [0, xp[-1]], color="#D55E00", lw=0.25)
            ax.plot([0, xp[-1]], [v, v], color="#D55E00", lw=0.25)
        ax.axvspan(G.GEOM["std"]["r_out"] * 1e6, xp[-1], color="0.85", zorder=0)
        ax.axhspan(G.GEOM["std"]["r_out"] * 1e6, xp[-1], color="0.85", zorder=0)
        ax.set_title(
            f"Cartesian {'graded' if graded else 'uniform'}: {(len(xn) - 1) ** 2:,} cells"
        )
    for ax in axs:
        ax.plot(np.cos(th), np.sin(th), "k-", lw=1.2)
        for r in G.GEOM["std"]["radii"]:
            ax.plot(r * 1e6 * np.cos(th), r * 1e6 * np.sin(th), "k:", lw=0.8)
        ax.set_xlim(0, 3.0)
        ax.set_ylim(0, 3.0)
        ax.set_aspect("equal")
        ax.set_xlabel("x [µm]")
        ax.set_ylabel("y [µm]")
    fig.suptitle(
        f"One quadrant of each mesh at res {res} (h_in = {h * 1e9:.0f} nm, h_out = 2 h_in); "
        "cylinder (solid), extraction circles (dotted), CPML (grey)"
    )
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / "a_graded_meshes.png")
    plt.close(fig)


def fig_error(acc):
    fig, axs = plt.subplots(2, 2, figsize=(11.5, 8), constrained_layout=True)
    for row, pol in enumerate(("TM", "TE")):
        for arm, st in STYLE.items():
            rr = series_of(acc, arm, pol)
            kw = dict(color=st["color"], marker=st["marker"], ls=st["ls"], ms=5, lw=1)
            if rr:
                axs[row, 0].loglog(
                    [d["dofs"] for d in rr],
                    [d["err_rms"] for d in rr],
                    label=st["label"],
                    **kw,
                )
                axs[row, 1].loglog(
                    [d["dofs"] * d["steps"] for d in rr],
                    [d["err_rms"] for d in rr],
                    **kw,
                )
            done = {d["res"] for d in rr}
            pend = [r for r in G.RES if r not in done]
            if pend:
                axs[row, 0].text(0.99, 0.02 + 0.04 * list(STYLE).index(arm),
                                 f"{st['label']}: res {pend} pending", transform=axs[row, 0].transAxes,
                                 ha="right", fontsize=6, color=st["color"])  # fmt: skip
        for col, (xl, slope) in enumerate(
            (("field unknowns", -1), ("work: unknowns × time steps", -2 / 3))
        ):
            ax = axs[row, col]
            for t in TARGETS:
                ax.axhline(t, color="0.5", lw=0.7, ls=":")
            xl_, yl = ax.get_xlim(), ax.get_ylim()
            xx = np.geomspace(*xl_, 10)
            ax.loglog(xx, 0.5 * np.sqrt(yl[0] * yl[1]) * (xx / np.sqrt(xl_[0] * xl_[1])) ** slope,
                      "k:", lw=1, label=f"slope {slope:.2g} (2nd order)")  # fmt: skip
            ax.set_xlim(xl_)
            ax.set_ylim(yl)
            ax.set_xlabel(xl)
            ax.set_ylabel(f"{pol}: RMS rel. error of Q_sca, x = 2 to 10")
            ax.grid(True, which="both", alpha=0.3)
        axs[row, 0].legend(fontsize=6.5, loc="lower left")
        axs[row, 1].legend(fontsize=6.5, loc="lower left")
    fig.suptitle(
        "Error vs unknowns and vs work: graded vs uniform, triangles vs Cartesian (dotted: 1 % and 2 %)"
    )
    fig.savefig(FIG / "b_error_vs_unknowns.png")
    plt.close(fig)


def table_targets(acc, timing):
    cpu = cost_per_update(timing, acc)
    rows, bars = [], {}
    for pol in ("TM", "TE"):
        for arm in STYLE:
            rr = series_of(acc, arm, pol)
            if not rr:
                continue
            fam = "cart" if arm.startswith("cart") else "tri"
            for t in TARGETS:
                n = to_target(rr, "dofs", t)
                c = to_target(rr, "cells", t)
                w = to_target(rr, "work", t)
                if arm.endswith("uniform"):  # contended accuracy runs: per-update cost
                    ns = [d["run_s"] / (d["dofs"] * d["steps"]) for (a, p, _), d in timing.items()
                          if a == arm and p == pol]  # fmt: skip
                    sec = w * (np.median(ns) if ns else cpu[fam][0])
                else:  # uncontended runs: interpolate the measured run time
                    sec = to_target(rr, "run_s", t)
                rows.append((pol, arm, t, n, c, w, sec))
                bars[(pol, arm, t)] = n
    lines = [
        "| pol | arm | target RMS | unknowns | cells | unknowns × steps | 1-core time [s] (uncontended) |",
        "|---|---|---|---|---|---|---|",
    ]
    for pol, arm, t, n, c, w, sec in rows:
        lines.append(
            f"| {pol} | {arm} | {100 * t:.0f} % | {n:,.0f} | {c:,.0f} | {w:.3g} | {sec:,.0f} |"
        )
    lines += [
        "",
        "Seconds per (unknown × step), median (min-max, n) of uncontended Batch runs:",
        "",
    ]
    for fam, (m, lo, hi, k) in cpu.items():
        lines.append(
            f"- {fam}: {m * 1e9:.2f} ns ({lo * 1e9:.2f}-{hi * 1e9:.2f}, n = {k})"
        )
    return lines, bars


def fig_bars(bars):
    arms = [
        a
        for a in STYLE
        if any((p, a, t) in bars for p in ("TM", "TE") for t in TARGETS)
    ]
    fig, axs = plt.subplots(
        1, 2, figsize=(11.5, 4.2), constrained_layout=True, sharey=True
    )
    w = 0.38
    for ax, pol in zip(axs, ("TM", "TE")):
        for j, t in enumerate(TARGETS):
            vals = [bars.get((pol, a, t), np.nan) for a in arms]
            xs = np.arange(len(arms)) + (j - 0.5) * w
            ax.bar(xs, np.nan_to_num(vals), w, color=[STYLE[a]["color"] for a in arms],
                   alpha=0.55 if t == 0.02 else 1.0, edgecolor="k", lw=0.5,
                   label=f"{100 * t:.0f} % RMS")  # fmt: skip
            for x, v in zip(xs, vals):
                ax.text(x, v if np.isfinite(v) else 6e4, f"{v / 1e3:.0f}k" if np.isfinite(v) else "not reached",
                        ha="center", va="bottom", fontsize=6.5, rotation=90)  # fmt: skip
        ax.set_yscale("log")
        ax.set_ylim(5e4, 4e6)
        ax.set_xticks(np.arange(len(arms)))
        ax.set_xticklabels([a.replace("-", "\n", 1) for a in arms], fontsize=7)
        ax.set_title(
            f"{pol}: unknowns to reach RMS error 2 % (light) and 1 % (dark)", fontsize=9
        )
        ax.set_ylabel("field unknowns (log-log interpolation)")
        ax.grid(True, axis="y", which="both", alpha=0.3)
    fig.savefig(FIG / "c_unknowns_to_target.png")
    plt.close(fig)


def fig_qsca(acc, res=20):
    from wakis.analytic import cylinder_efficiencies

    xd = np.linspace(1.9, 10.1, 2000)
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4), constrained_layout=True)
    for ax, pol in zip(axs, ("TM", "TE")):
        ax.plot(
            xd,
            cylinder_efficiencies(S.N_CYL, xd, pol)[1],
            "k-",
            lw=1,
            label="exact series",
        )
        for arm in ("cart-graded", "tri-graded", "tri-graded-compact"):
            d = acc.get((arm, pol, res))
            if d:
                st = STYLE[arm]
                ax.plot(S.X, d["q"], ls="none", marker=st["marker"], mfc="none", ms=4,
                        color=st["color"], label=st["label"])  # fmt: skip
        ax.set_xlabel("size parameter x = 2πa/λ")
        ax.set_ylabel("Q_sca")
        ax.set_title(f"{pol}, res {res}")
        ax.legend(fontsize=7)
    fig.savefig(FIG / "d_qsca_graded.png")
    plt.close(fig)


def main_plot():
    FIG.mkdir(exist_ok=True)
    acc, timing = load()
    lines = [
        "| arm | pol | res | h_in [nm] | unknowns | cells | dt [as] | steps | unknowns × steps | run [s] | RMS err | max err | max refl. |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for pol in ("TM", "TE"):
        for arm in STYLE:
            for d in series_of(acc, arm, pol):
                lines.append(
                    f"| {arm} | {pol} | {d['res']} | {d['h'] * 1e9:.1f} | {d['dofs']:,} | {d['cells']:,} "
                    f"| {d['dt'] * 1e18:.1f} | {d['steps']} | {d['dofs'] * d['steps']:.3g} | {d['run_s']:.0f} "
                    f"| {100 * d['err_rms']:.2f} % | {100 * d['err_max']:.2f} % | {d['refl_max']:.3f} |"
                )
    lines += [
        "",
        "(uniform arms: accuracy runs of ../mie-cylinder-mesh, run times there were contended)",
        "",
    ]
    lines += ["Uncontended timing runs (one run per n2-standard-2):", "",
              "| arm | pol | res | unknowns | steps | run [s] | ns per unknown-step |", "|---|---|---|---|---|---|---|"]  # fmt: skip
    for (arm, pol, res), d in sorted(timing.items()):
        lines.append(f"| {arm} | {pol} | {res} | {d['dofs']:,} | {d['steps']} | {d['run_s']:.0f} "
                     f"| {d['run_s'] / (d['dofs'] * d['steps']) * 1e9:.2f} |")  # fmt: skip
    tl, bars = table_targets(acc, timing)
    lines += [
        "",
        "Unknowns / work / time to reach a target RMS error (log-log interpolation):",
        "",
    ] + tl
    txt = "\n".join(lines)
    (DATA / "summary.md").write_text(txt + "\n")
    print(txt)
    fig_error(acc)
    fig_bars(bars)
    fig_qsca(acc)
    fig_meshes()


if __name__ == "__main__":
    main_plot()
