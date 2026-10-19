"""Figures and tables of the subgridding Mie study (B1).

Reads data/sg*.json of this study and, for the baselines, data/baselines/
(the graded-study runs of branch feature/tri-fixes, commit 75f603d: graded
Cartesian and graded triangles, including the radial-PML arm)."""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "mie-cylinder-mesh"))
sys.path.insert(0, str(HERE.parent / "mie-cylinder-graded"))
import plot_graded as PG  # noqa: E402

DATA = HERE / "data"
FIG = HERE / "figures"
TARGETS = (0.02, 0.01)
BASE = ["cart-graded", "cart-graded-compact", "tri-graded", "tri-graded-compact",
        "tri-graded-pml"]  # fmt: skip
STYLE = {a: PG.STYLE[a] for a in BASE if a in PG.STYLE}
STYLE["tri-graded-pml"] = dict(color="#56B4E9", marker="d", ls="-",
                               label="Wakis FIT (triangles graded, radial PML)")  # fmt: skip
STYLE.update({
    "sg": dict(color="#000000", marker="P", ls="-", label="Wakis FIT (Cartesian 2:1 subgrid)"),
    "sg-compact": dict(color="#7F7F7F", marker="X", ls="-", label="Wakis FIT (Cartesian 2:1 subgrid, compact domain)"),
    "sg-lts": dict(color="#000000", marker="P", ls=":", label="Wakis FIT (Cartesian 2:1 subgrid + LTS p=2)"),
    "sg-compact-lts": dict(color="#7F7F7F", marker="X", ls=":", label="Wakis FIT (subgrid compact + LTS p=2)"),
})  # fmt: skip
plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 150, "font.size": 9})


def load():
    acc = {}
    files = sorted((DATA / "baselines").glob("*.json")) + sorted(DATA.glob("sg*.json"))
    for f in files:
        d = json.loads(f.read_text())
        if "error" in d:
            print("FAILED", f.name, d["error"])
            continue
        PG._metrics(d)
        lts = d.get("lts")
        # effective work: LTS does (p - 1) extra products on the fine set F
        frac = lts["fine_fraction"] if lts else 0.0
        d["work_eff"] = float(d["dofs"]) * d["steps"] * (1 + frac)
        if d["arm"] in BASE or d["arm"].startswith("sg"):
            acc[(d["arm"], d["pol"], d["res"])] = d
    for d in acc.values():
        d.setdefault("work_eff", float(d["dofs"]) * d["steps"])
    return acc


def to_target(rr, key, target):
    if key == "work_eff":
        rr = [dict(d, _w=d["work_eff"]) for d in rr]
        return PG.to_target(rr, "_w", target)
    return PG.to_target(rr, key, target)


def table(acc):
    lines = [
        "| pol | arm | target RMS | unknowns | cells | unknowns × steps | "
        "1-core time [s] (uncontended) |",
        "|---|---|---|---|---|---|---|",
    ]
    res = {}
    for pol in ("TM", "TE"):
        for arm in STYLE:
            rr = PG.series_of(acc, arm, pol)
            if not rr:
                continue
            for t in TARGETS:
                n = to_target(rr, "dofs", t)
                c = to_target(rr, "cells", t)
                w = to_target(rr, "work_eff", t)
                s = to_target(rr, "run_s", t)
                res[(pol, arm, t)] = (n, c, w, s)
                lines.append(f"| {pol} | {arm} | {100 * t:.0f} % | {n:,.0f} | {c:,.0f} | "
                             f"{w:.3g} | {s:,.0f} |")  # fmt: skip
    return lines, res


def runs_table(acc):
    lines = [
        "| arm | pol | res | h_in [nm] | unknowns | cells (coarse + fine) | dt [as] | steps "
        "| work | run [s] | ns/unknown-step | RMS err | max err | max refl. | LTS F |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for (arm, pol, r), d in sorted(
        acc.items(), key=lambda kv: (kv[0][1], kv[0][0], kv[0][2])
    ):
        if not arm.startswith("sg"):
            continue
        lts = d.get("lts")
        lines.append(
            f"| {arm} | {pol} | {r} | {d['h'] * 1e9:.1f} | {d['dofs']:,} | "
            f"{d['cells']:,} ({d['cells_coarse']:,} + {d['cells_fine']:,}) | "
            f"{d['dt'] * 1e18:.1f} | {d['steps']} | {d['work_eff']:.3g} | {d['run_s']:.0f} | "
            f"{d['run_s'] / (d['dofs'] * d['steps']) * 1e9:.1f} | {100 * d['err_rms']:.2f} % | "
            f"{100 * d['err_max']:.2f} % | {d['refl_max']:.3f} | "
            f"{'%.1f %%' % (100 * lts['fine_fraction']) if lts else '-'} |"
        )  # fmt: skip
    return lines


def fig_error(acc):
    fig, axs = plt.subplots(2, 2, figsize=(10, 7.5), sharey=True)
    for i, pol in enumerate(("TM", "TE")):
        for arm, st in STYLE.items():
            rr = PG.series_of(acc, arm, pol)
            if not rr:
                continue
            e = [d["err_rms"] for d in rr]
            kw = dict(color=st["color"], marker=st["marker"], ls=st["ls"], ms=5)
            axs[i, 0].loglog([d["dofs"] for d in rr], e, label=st["label"], **kw)
            axs[i, 1].loglog([d["work_eff"] for d in rr], e, **kw)
        for ax in axs[i]:
            for t in TARGETS:
                ax.axhline(t, color="0.6", lw=0.6, ls="--")
            ax.grid(True, which="both", alpha=0.25)
        axs[i, 0].set_ylabel(f"{pol}: RMS error of Q_sca vs exact series")
    axs[1, 0].set_xlabel("field unknowns")
    axs[1, 1].set_xlabel("work = unknowns × steps (LTS: × (1 + fine fraction))")
    axs[0, 0].legend(fontsize=7, loc="lower left")
    fig.suptitle(
        "Mie cylinder n = 2, x = 2-10: subgridded Cartesian vs graded baselines"
    )
    fig.tight_layout()
    fig.savefig(FIG / "b_error_vs_cost.png")
    plt.close(fig)


def fig_bars(res):
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    arms = [a for a in STYLE]
    for ax, key, lab in (
        (axs[0], 0, "field unknowns"),
        (axs[1], 2, "work (unknowns × steps)"),
    ):
        x = np.arange(len(arms))
        for k, (pol, off) in enumerate((("TM", -0.2), ("TE", 0.2))):
            for t, alpha in ((0.02, 0.45), (0.01, 1.0)):
                v = [res.get((pol, a, t), (np.nan,) * 4)[key] for a in arms]
                ax.bar(x + off, v, width=0.38, color=[STYLE[a]["color"] for a in arms],
                       alpha=alpha, edgecolor="k", lw=0.4,
                       hatch="//" if pol == "TE" else None)  # fmt: skip
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(arms, rotation=30, ha="right", fontsize=7)
        ax.set_ylabel(lab + " to reach target")
        ax.grid(True, axis="y", which="both", alpha=0.25)
    axs[0].set_title("light: 2 %, dark: 1 % RMS; plain TM, hatched TE", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "c_cost_to_target.png")
    plt.close(fig)


def fig_mesh(res=10):
    import subgrid_study as SS

    mesh, solver, h = SS.build("compact", "TM", res)
    c = mesh.cells
    fig, ax = plt.subplots(figsize=(5.2, 5))
    sel = (c["x1"] > -1e-9) & (c["y1"] > -1e-9)
    for k in np.flatnonzero(sel):
        ax.add_patch(plt.Rectangle((c["x0"][k] * 1e6, c["y0"][k] * 1e6),
                                   (c["x1"][k] - c["x0"][k]) * 1e6,
                                   (c["y1"][k] - c["y0"][k]) * 1e6, fill=False,
                                   lw=0.3, ec="0.3" if c["level"][k] else "0.6"))  # fmt: skip
    hang = mesh.nodes[mesh.hanging]
    hang = hang[(hang[:, 0] >= 0) & (hang[:, 1] >= 0)]
    ax.plot(*(hang.T * 1e6), "o", ms=1.5, color="#D55E00", label="hanging nodes")
    th = np.linspace(0, np.pi / 2, 200)
    ax.plot(np.cos(th), np.sin(th), color="#0072B2", lw=1.5, label="cylinder a = 1 um")
    for r in SS.GEOM["compact"]["radii"]:
        ax.plot(
            r * 1e6 * np.cos(th), r * 1e6 * np.sin(th), "--", lw=0.6, color="#009E73"
        )
    ax.axvline(1.7, color="#CC79A7", lw=1, label="PML start (1.7 um)")
    ax.axhline(1.7, color="#CC79A7", lw=1)
    lc = mesh.level_counts()
    ax.set_title(f"res {res}: {lc[0]:,} coarse + {lc[1]:,} fine cells, "
                 f"{solver.n_unknowns:,} unknowns", fontsize=8)  # fmt: skip
    ax.set_xlim(0, mesh.xc[-1] * 1e6)
    ax.set_ylim(0, mesh.xc[-1] * 1e6)
    ax.set_aspect("equal")
    ax.set_xlabel("x [um]")
    ax.set_ylabel("y [um]")
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "a_subgrid_mesh.png")
    plt.close(fig)


def fig_qsca(acc, res=20):
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    from mie_mesh_study import X

    for ax, pol in zip(axs, ("TM", "TE")):
        d = acc.get(("sg-compact", pol, res))
        if d is None:
            continue
        ax.plot(X, d["q_ref"], "-", color="k", lw=1.2, label="exact series")
        for arm in ("sg-compact", "sg-compact-lts"):
            dd = acc.get((arm, pol, res))
            if dd:
                st = STYLE[arm]
                ax.plot(X, dd["q"], st["marker"], color=st["color"], ms=4,
                        mfc="none", label=f"{st['label']}, res {res}")  # fmt: skip
        ax.set_xlabel("size parameter x = k a")
        ax.set_ylabel(f"Q_sca ({pol})")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "d_qsca_subgrid.png")
    plt.close(fig)


def main_plot():
    FIG.mkdir(exist_ok=True)
    acc = load()
    lines, res = table(acc)
    fig_error(acc)
    fig_bars(res)
    fig_qsca(acc)
    try:
        fig_mesh()
    except Exception as e:  # noqa: BLE001
        print("mesh figure skipped:", e)
    out = ["Per-run results (this study):", ""] + runs_table(acc) + [
        "", "Unknowns / work / time to reach a target RMS error (log-log interpolation; "
        "baselines from ../mie-cylinder-graded):", ""] + lines  # fmt: skip
    (DATA / "summary.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main_plot()
