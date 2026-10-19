"""Figures and tables of the polygonal-FIT Mie study.

Reads data/*.json (this study) and ../mie-cylinder-graded/data/*.json (the
graded Cartesian and triangle baselines, same problem, extraction and
machine type: n2-standard-2, one run per machine).

    python plot_polyfit.py
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "mie-cylinder-mesh"))
sys.path.insert(0, str(HERE.parent / "mie-cylinder-graded"))

DATA = HERE / "data"
FIG = HERE / "figures"
BASE = HERE.parent / "mie-cylinder-graded" / "data"
STYLE = {
    "cart-graded": dict(color="#D55E00", marker="s", ls=":", label="Wakis FIT (Cartesian graded, conformal; square + CPML)"),
    "tri-graded": dict(color="#0072B2", marker="o", ls=":", label="Wakis FIT (triangles graded; disk + ABC)"),
    "hybrid": dict(color="#E69F00", marker="D", ls="-", label="Wakis FIT (hybrid squares + triangles; square + PML)"),
    "hybrid-disk": dict(color="#009E73", marker="P", ls="-", label="Wakis FIT (hybrid squares + triangles; disk + ABC)"),
    "voronoi": dict(color="#CC79A7", marker="h", ls="-", label="Wakis FIT (Voronoi polygons; disk + ABC)"),
    "cart-graded-compact": dict(color="#D55E00", marker="s", ls=":", mfc="none", label="Wakis FIT (Cartesian graded, compact)"),
    "tri-graded-compact": dict(color="#0072B2", marker="o", ls=":", mfc="none", label="Wakis FIT (triangles graded, compact)"),
    "hybrid-compact": dict(color="#E69F00", marker="D", ls="-", mfc="none", label="Wakis FIT (hybrid, square + PML, compact)"),
    "hybrid-disk-compact": dict(color="#009E73", marker="P", ls="-", mfc="none", label="Wakis FIT (hybrid, disk + ABC, compact)"),
    "voronoi-compact": dict(color="#CC79A7", marker="h", ls="-", mfc="none", label="Wakis FIT (Voronoi, compact)"),
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
    acc = {}
    for folder in (BASE, DATA):
        for f in sorted(folder.glob("*.json")):
            d = json.loads(f.read_text())
            if "error" in d or d.get("arm") not in STYLE:
                continue
            if not np.all(np.isfinite(d["q"])):
                print("NaN result", f.name)
                continue
            acc[(d["arm"], d["pol"], d["res"])] = _metrics(d)
    return acc


def series_of(acc, arm, pol):
    return sorted((d for (a, p, _), d in acc.items() if a == arm and p == pol),
                  key=lambda d: d["res"])  # fmt: skip


def _val(d, key):
    return float(d["dofs"]) * d["steps"] if key == "work" else float(d[key])


def to_target(rr, key, target):
    """log-log interpolation of `key` where the RMS error crosses `target`
    (identical to ../mie-cylinder-graded/plot_graded.py)."""
    e = np.array([d["err_rms"] for d in rr])
    v = np.array([_val(d, key) for d in rr])
    for i in range(len(rr) - 1):
        if e[i] >= target >= e[i + 1]:
            t = (np.log(target) - np.log(e[i])) / (np.log(e[i + 1]) - np.log(e[i]))
            return float(np.exp(np.log(v[i]) + t * (np.log(v[i + 1]) - np.log(v[i]))))
    return np.nan


# --------------------------------------------------------------------------
def _draw(ax, nodes, polys, tags, lim, title):
    sel = [i for i, p in enumerate(polys) if p.min(0).min() > -0.05e-6 and p.max(0).max() < lim]
    pc = PolyCollection(
        [polys[i] * 1e6 for i in sel],
        facecolors=np.where(np.asarray(tags)[sel] == 0, "#f4c99a", "#cfe3f5"),
        edgecolors="k", linewidths=0.25,
    )  # fmt: skip
    ax.add_collection(pc)
    th = np.linspace(0, np.pi / 2, 200)
    ax.plot(np.cos(th), np.sin(th), "r-", lw=0.8)
    ax.set_xlim(0, lim * 1e6)
    ax.set_ylim(0, lim * 1e6)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=8)
    ax.set_xlabel("x [µm]")


def fig_meshes(res=10):
    import polyfit_study as P

    from wakis.unstructured import graded_disk_mesh
    from wakis.unstructured.polymesh import PolyMesh

    sys.path.insert(0, str(HERE.parent / "mie-cylinder-graded"))
    import graded_study as GS

    h = P.S.LAM_MIN_DIEL / res
    panels = []
    tri = PolyMesh.from_simplex(graded_disk_mesh([P.A, 2.7e-6], GS.size_field(h, True)))
    panels.append((tri, f"triangles graded ({tri.n_edges + tri.n_nodes:,} unknowns)"))
    for kind in ("hybrid", "hybrid-disk", "voronoi"):
        m, info, kw, _ = P.build(kind, "std", res)
        panels.append((m, f"{kind} ({m.n_nodes + m.n_edges:,} unknowns)"))
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.6), sharey=True)
    for ax, (m, t) in zip(axs, panels):
        polys = [m.nodes[m.cell(i)] for i in range(m.n_cells)]
        _draw(ax, m.nodes, polys, m.cell_tags, 2.0e-6, t)
    axs[0].set_ylabel("y [µm]")
    fig.suptitle(f"Mesh quadrants at res {res} (h_in = λ_min,diel / {res}); cylinder n = 2 (orange), r = a in red; whole-domain unknowns", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "a_meshes.png")
    plt.close(fig)


def fig_error(acc):
    fig, axs = plt.subplots(2, 2, figsize=(11, 8))
    for i, pol in enumerate(("TM", "TE")):
        for j, key in enumerate(("dofs", "work")):
            ax = axs[i, j]
            for arm, st in STYLE.items():
                rr = series_of(acc, arm, pol)
                if not rr:
                    continue
                x = [_val(d, key) for d in rr]
                y = [100 * d["err_rms"] for d in rr]
                ax.loglog(x, y, color=st["color"], marker=st["marker"], ls=st["ls"],
                          mfc=st.get("mfc", st["color"]), ms=5, lw=1, label=st["label"])  # fmt: skip
            for t in TARGETS:
                ax.axhline(100 * t, color="0.6", lw=0.6, ls="--")
            x0 = 1e5 if key == "dofs" else 3e9
            ax.loglog([x0, 8 * x0], [4, 4 / 8 ** (1.0 if key == "dofs" else 2 / 3)], "k:", lw=0.8)
            ax.text(8 * x0, 4 / 8 ** (1.0 if key == "dofs" else 2 / 3), "  2nd order", fontsize=7)
            ax.set_xlabel("field unknowns" if key == "dofs" else "work = unknowns × steps")
            ax.set_ylabel(f"{pol}: RMS error of Q_sca [%]")
            ax.grid(True, which="both", alpha=0.2)
    axs[0, 0].legend(fontsize=6, loc="lower left")
    fig.suptitle("Mie cylinder n = 2, x = 2-10: RMS error vs exact series (filled: standard domains, open: compact)", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "b_error_vs_cost.png")
    plt.close(fig)


def table_targets(acc):
    rows = []
    for pol in ("TM", "TE"):
        for arm in STYLE:
            rr = series_of(acc, arm, pol)
            if not rr:
                continue
            for t in TARGETS:
                rows.append((pol, arm, t, to_target(rr, "dofs", t), to_target(rr, "cells", t),
                             to_target(rr, "work", t), to_target(rr, "run_s", t)))  # fmt: skip
    return rows


def fig_targets(rows):
    arms = [a for a in STYLE if any(r[1] == a for r in rows)]
    fig, axs = plt.subplots(2, 3, figsize=(13, 6.5))
    for i, pol in enumerate(("TM", "TE")):
        for j, (k, lab) in enumerate(((3, "unknowns"), (5, "work (unknowns × steps)"), (6, "uncontended 1-core time [s]"))):
            ax = axs[i, j]
            for m, t in enumerate(TARGETS):
                vals = [next((r[k] for r in rows if r[0] == pol and r[1] == a and r[2] == t), np.nan) for a in arms]
                xs = np.arange(len(arms)) + (m - 0.5) * 0.38
                ax.bar(xs, vals, width=0.36, color=[STYLE[a]["color"] for a in arms],
                       alpha=0.45 if t == 0.02 else 1.0, edgecolor="k", lw=0.4,
                       label=f"{100 * t:.0f} % RMS")  # fmt: skip
            ax.set_yscale("log")
            ax.set_xticks(np.arange(len(arms)))
            ax.set_xticklabels(arms, rotation=40, ha="right", fontsize=7)
            ax.set_ylabel(f"{pol}: {lab}")
            ax.grid(True, axis="y", which="both", alpha=0.2)
    axs[0, 0].legend(fontsize=7)
    fig.suptitle("Cost to reach 2 % (light) and 1 % (dark) RMS error (log-log interpolation; missing bar = target not reached)", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "c_cost_to_target.png")
    plt.close(fig)


def fig_qsca(acc, res=20):
    import mie_mesh_study as S

    xs = np.linspace(2, 10, 600)
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, pol in zip(axs, ("TM", "TE")):
        q_ref, _ = S.series(pol, xs)
        ax.plot(xs, q_ref, "k-", lw=1, label="exact series")
        for arm in ("hybrid-disk", "voronoi", "tri-graded"):
            d = acc.get((arm, pol, res))
            if d is None:
                continue
            st = STYLE[arm]
            ax.plot(S.X, d["q"], st["marker"], color=st["color"], ms=4, mfc="none",
                    label=f"{st['label']} {d['dofs']:,} unknowns, {100 * d['err_rms']:.2f} %")  # fmt: skip
        ax.set_xlabel("size parameter x = k a")
        ax.set_ylabel(f"{pol}: Q_sca")
        ax.legend(fontsize=6)
    fig.suptitle(f"Q_sca at res {res}", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "d_qsca.png")
    plt.close(fig)


def summary(acc, rows):
    L = ["| arm | pol | res | h_in [nm] | unknowns | cells | dt [as] | steps | unknowns × steps | run [s] | ns/(unknown·step) | RMS err | max err | max refl. |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for (arm, pol, res), d in sorted(acc.items(), key=lambda kv: (kv[0][1], kv[0][0], kv[0][2])):
        w = d["dofs"] * d["steps"]
        L.append(
            f"| {arm} | {pol} | {res} | {d['h'] * 1e9:.1f} | {d['dofs']:,} | {d['cells']:,} | {d['dt'] * 1e18:.1f} | {d['steps']} | {w:.3g} | {d['run_s']:.0f} | {1e9 * d['run_s'] / w:.1f} | {100 * d['err_rms']:.2f} % | {100 * d['err_max']:.2f} % | {d['refl_max']:.3f} |"
        )
    L += ["", "Cost to reach a target RMS error (log-log interpolation between runs):", "",
          "| pol | arm | target | unknowns | cells | unknowns × steps | 1-core time [s] |",
          "|---|---|---|---|---|---|---|"]  # fmt: skip
    for pol, arm, t, n, c, w, s in rows:
        L.append(f"| {pol} | {arm} | {100 * t:.0f} % | {n:,.0f} | {c:,.0f} | {w:.3g} | {s:,.0f} |")
    (DATA / "summary.md").write_text("\n".join(L) + "\n")


def main():
    FIG.mkdir(exist_ok=True)
    acc = load()
    rows = table_targets(acc)
    summary(acc, rows)
    fig_error(acc)
    fig_targets(rows)
    fig_qsca(acc)
    if "--no-mesh" not in sys.argv:
        fig_meshes()


if __name__ == "__main__":
    main()
