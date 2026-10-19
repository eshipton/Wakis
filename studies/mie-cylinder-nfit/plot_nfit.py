"""Figures and tables of the curvilinear-NFIT Mie study (data/*.json) against
the baselines of ../mie-cylinder-graded (copied to data/baselines/, Batch runs
of branch feature/tri-fixes 75f603d)."""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "mie-cylinder-mesh"))
sys.path.insert(0, str(HERE))
import mie_mesh_study as S  # noqa: E402

DATA = HERE / "data"
FIG = HERE / "figures"
TARGETS = (0.02, 0.01)
STYLE = {
    "ogrid": dict(color="#D62728", marker="o", ls="-", label="Wakis FIT (O-grid NFIT, radial PML)"),
    "ogrid-corner": dict(color="#FF9896", marker="x", ls="--", label="Wakis FIT (O-grid, corner Hodge everywhere)"),
    "ogrid-std": dict(color="#9467BD", marker="d", ls=":", label="Wakis FIT (O-grid, baseline domain r = 2.7 um)"),
    "multiblock": dict(color="#8C564B", marker="P", ls="-.", label="Wakis FIT (O-grid + Cartesian multi-block)"),
    "tri-graded-pml": dict(color="#000000", marker="*", ls="-", label="Wakis FIT (triangles graded, radial PML)"),
    "tri-graded": dict(color="#0072B2", marker="^", ls=":", label="Wakis FIT (triangles graded, ABC)"),
    "cart-graded-compact": dict(color="#CC79A7", marker="v", ls="-", label="Wakis FIT (Cartesian graded conformal, compact)"),
    "cart-graded": dict(color="#E69F00", marker="s", ls=":", label="Wakis FIT (Cartesian graded conformal)"),
    "shell-ogrid": dict(color="#D62728", marker="o", ls="-", label="Wakis FIT (O-grid NFIT, radial refinement)"),
    "shell-tri": dict(color="#000000", marker="*", ls="-", label="Wakis FIT (triangles graded, isotropic refinement)"),
    "shell-cart": dict(color="#CC79A7", marker="v", ls="-", label="Wakis FIT (Cartesian graded, sub-cell shell)"),
}  # fmt: skip
plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 150, "font.size": 9})


def _metrics(d):
    if d.get("q_ref_missing") or d["arm"].startswith("shell"):
        from shell_study import reference

        d["q_ref"] = reference(d["pol"])[0].tolist()
    q, r = np.array(d["q"]), np.array(d["q_ref"])
    d["err_rms"] = float(np.sqrt(np.mean((q / r - 1) ** 2)))
    d["err_max"] = float(np.max(np.abs(q / r - 1)))
    d["refl_max"] = float(np.max(d["refl"]))
    d["work"] = float(d["dofs"]) * d["steps"]
    return d


def load():
    acc = {}
    for f in sorted(list(DATA.glob("*.json")) + list((DATA / "baselines").glob("*.json"))):
        d = json.loads(f.read_text())
        if "error" in d:
            print("FAILED", f.name, d["error"])
            continue
        _metrics(d)
        acc[(d["arm"], d["pol"], d["res"])] = d
    return acc


def series_of(acc, arm, pol):
    return [acc[k] for k in sorted(k for k in acc if k[0] == arm and k[1] == pol)]


def to_target(rr, key, target):
    """log-log interpolation of `key` where the RMS error crosses `target`."""
    rr = sorted(rr, key=lambda d: d["dofs"])
    e = np.array([d["err_rms"] for d in rr])
    v = np.array([float(d[key]) for d in rr])
    for i in range(len(rr) - 1):
        if e[i] >= target >= e[i + 1]:
            t = (np.log(target) - np.log(e[i])) / (np.log(e[i + 1]) - np.log(e[i]))
            return float(np.exp(np.log(v[i]) + t * (np.log(v[i + 1]) - np.log(v[i]))))
    return np.nan


# --------------------------------------------------------------------------
def _draw(ax, mesh, color="k", lw=0.3, quadrant=True):
    from matplotlib.collections import LineCollection

    X = mesh.nodes * 1e6
    segs = X[mesh.edges]
    if quadrant:
        keep = (segs[..., 0].min(1) >= -1e-9) & (segs[..., 1].min(1) >= -1e-9)
        segs = segs[keep]
    ax.add_collection(LineCollection(segs, colors=color, linewidths=lw))


def fig_meshes(res=10):
    import nfit_study as NS

    fig, axs = plt.subplots(1, 3, figsize=(13, 4.3), constrained_layout=True)
    th = np.linspace(0, np.pi / 2, 200)
    for ax, arm, title in ((axs[0], "ogrid", "single-block O-grid + radial PML"),
                           (axs[1], "multiblock", "O-grid + transition + Cartesian block")):  # fmt: skip
        mesh, pml, radii, h = NS.build(arm, res)
        _draw(ax, mesh)
        gen = ~mesh.is_polar
        from matplotlib.collections import PolyCollection

        P = mesh.nodes[mesh.cells[gen]] * 1e6
        ax.add_collection(PolyCollection(P, facecolors="#fdd0a2", edgecolors="none", alpha=0.7, zorder=0))
        ax.plot(np.cos(th), np.sin(th), color="#0072B2", lw=1.2)
        for r in radii:
            ax.plot(r * 1e6 * np.cos(th), r * 1e6 * np.sin(th), color="#2CA02C", lw=0.8, ls="--")
        r0 = pml["r0"] * 1e6
        if pml["kind"] == "radial":
            ax.plot(r0 * np.cos(th), r0 * np.sin(th), color="#D62728", lw=1.0)
            lim = pml["r1"] * 1e6
        else:
            ax.plot([r0, r0, 0], [0, r0, r0], color="#D62728", lw=1.0)
            lim = mesh.nodes.max() * 1e6
        ax.set_xlim(0, lim * 1.02)
        ax.set_ylim(0, lim * 1.02)
        ax.set_aspect("equal")
        ax.set_title(f"{title}\nres {res}: {mesh.n_cells:,} cells", fontsize=9)
        ax.set_xlabel("x [um]")
        ax.set_ylabel("y [um]")
    # shell zoom
    mesh, pml, radii, h = NS.build("ogrid", res, shell=(20e-9, 12.25))
    ax = axs[2]
    _draw(ax, mesh, lw=0.4, quadrant=False)
    for r, c in ((1.0, "#0072B2"), (1.02, "#D62728")):
        ax.plot(r * np.cos(th), r * np.sin(th), color=c, lw=1.0)
    ax.set_xlim(0.85, 1.15)
    ax.set_ylim(-0.05, 0.25)
    ax.set_aspect("equal")
    ax.set_title("coated cylinder (20 nm shell): radial-only refinement", fontsize=9)
    ax.set_xlabel("x [um]")
    ax.set_ylabel("y [um]")
    fig.suptitle("Body-fitted meshes (orange: nonorthogonal corner-Hodge cells; white: polar/rectangular "
                 "diagonal cells; green: extraction rings; red: PML start)", fontsize=9)  # fmt: skip
    fig.savefig(FIG / "a_meshes.png")
    plt.close(fig)


def fig_error(acc, arms, name, title):
    fig, axs = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for j, pol in enumerate(("TM", "TE")):
        for i, key in enumerate(("dofs", "work")):
            ax = axs[i, j]
            for arm in arms:
                rr = sorted(series_of(acc, arm, pol), key=lambda d: d["dofs"])
                if not rr:
                    continue
                st = STYLE[arm]
                ax.loglog([d[key] for d in rr], [100 * d["err_rms"] for d in rr],
                          color=st["color"], marker=st["marker"], ls=st["ls"], label=st["label"])  # fmt: skip
            for t in TARGETS:
                ax.axhline(100 * t, color="0.6", lw=0.6, ls="--")
            ax.set_xlabel({"dofs": "field unknowns", "work": "work = unknowns x steps"}[key])
            ax.set_ylabel("RMS rel. error of Q_sca [%]")
            ax.set_title(f"{pol}", fontsize=9)
            ax.grid(True, which="both", lw=0.3)
            x0 = np.array(ax.get_xlim())
            ax.loglog(x0, 100 * 0.02 * (x0 / x0[0]) ** -1.0 * 3, color="0.7", lw=0.6, ls=":",
                      label="2nd order (error ~ unknowns^-1)")  # fmt: skip
    axs[0, 0].legend(fontsize=7)
    fig.suptitle(title, fontsize=10)
    fig.savefig(FIG / name)
    plt.close(fig)


def fig_qsca(acc, res=20):
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)
    for ax, pol in zip(axs, ("TM", "TE")):
        d = acc.get(("ogrid", pol, res))
        if d is None:
            continue
        xf = np.linspace(2, 10, 801)
        from wakis.analytic import cylinder_efficiencies

        ax.plot(xf, cylinder_efficiencies(S.N_CYL, xf, pol)[1], "k-", lw=1, label="exact series")
        ax.plot(S.X, d["q"], "o", color=STYLE["ogrid"]["color"], ms=4,
                label=f"Wakis FIT (O-grid NFIT), res {res}, RMS {100 * d['err_rms']:.2f} %")  # fmt: skip
        ax.set_xlabel("size parameter x = k a")
        ax.set_ylabel("Q_sca")
        ax.set_title(f"{pol}: n = 2 cylinder", fontsize=9)
        ax.legend(fontsize=8)
    fig.savefig(FIG / "c_qsca_ogrid.png")
    plt.close(fig)


def fig_shell_qsca(acc, res=20):
    from shell_study import reference

    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)
    for ax, pol in zip(axs, ("TM", "TE")):
        ax.plot(S.X, reference(pol)[0], "k-", lw=1, label="exact series (coated)")
        from wakis.analytic import cylinder_efficiencies

        ax.plot(S.X, cylinder_efficiencies(S.N_CYL, S.X, pol)[1] * 1.0 / 1.02, color="0.6",
                lw=0.8, ls="--", label="exact series, no shell")  # fmt: skip
        for arm in ("shell-ogrid", "shell-tri", "shell-cart"):
            d = acc.get((arm, pol, res))
            if d is None:
                continue
            st = STYLE[arm]
            ax.plot(S.X, d["q"], st["marker"], color=st["color"], ms=4, mfc="none",
                    label=f"{st['label']}, RMS {100 * d['err_rms']:.1f} %")  # fmt: skip
        ax.set_xlabel("x = k a (a = 1 um core)")
        ax.set_ylabel("Q_sca (normalized by 2 (a + t))")
        ax.set_title(f"{pol}: n = 2 core + 20 nm n = 3.5 shell, res {res}", fontsize=9)
        ax.legend(fontsize=7)
    fig.savefig(FIG / "e_shell_qsca.png")
    plt.close(fig)


def table(acc, arms):
    lines = ["| pol | arm | unknowns 2 % | unknowns 1 % | work 2 % | work 1 % | run time 1 % [s] |",
             "|---|---|---|---|---|---|---|"]  # fmt: skip
    rows = {}
    for pol in ("TM", "TE"):
        for arm in arms:
            rr = series_of(acc, arm, pol)
            if not rr:
                continue
            v = [to_target(rr, k, t) for t in TARGETS for k in ("dofs", "work")]
            ts = to_target(rr, "run_s", 0.01)
            rows[(arm, pol)] = dict(u2=v[0], w2=v[1], u1=v[2], w1=v[3], t1=ts)
            f = lambda x: "not reached" if np.isnan(x) else f"{x:,.0f}"  # noqa: E731
            g = lambda x: "not reached" if np.isnan(x) else f"{x:.2e}"  # noqa: E731
            lines.append(f"| {pol} | {arm} | {f(v[0])} | {f(v[2])} | {g(v[1])} | {g(v[3])} | {f(ts)} |")
    return lines, rows


def runs_table(acc, arms):
    lines = ["| arm | pol | res | unknowns | cells | dt [as] | c dt / h_in | steps | work | run [s] | ns/(unknown step) | RMS err | max err | max refl |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for arm in arms:
        for pol in ("TM", "TE"):
            for d in sorted(series_of(acc, arm, pol), key=lambda d: d["res"]):
                cdt = d["dt"] * S.C0 / d["h"]
                lines.append(
                    f"| {arm} | {pol} | {d['res']} | {d['dofs']:,} | {d['cells']:,} | {d['dt'] * 1e18:.1f} | "
                    f"{cdt:.3f} | {d['steps']} | {d['work']:.2e} | {d['run_s']:.0f} | "
                    f"{1e9 * d['run_s'] / d['work']:.1f} | {100 * d['err_rms']:.2f} % | "
                    f"{100 * d['err_max']:.2f} % | {d['refl_max']:.3f} |")  # fmt: skip
    return lines


def main_plot():
    FIG.mkdir(exist_ok=True)
    acc = load()
    main_arms = ["ogrid", "ogrid-corner", "ogrid-std", "multiblock", "tri-graded-pml",
                 "cart-graded-compact", "tri-graded", "cart-graded"]  # fmt: skip
    shell_arms = ["shell-ogrid", "shell-tri", "shell-cart"]
    fig_meshes()
    fig_error(acc, main_arms, "b_error_vs_cost.png",
              "Mie cylinder n = 2, x = 2-10: Q_sca RMS error vs cost")  # fmt: skip
    fig_qsca(acc)
    fig_error(acc, shell_arms, "d_shell_error_vs_cost.png",
              "Coated cylinder (20 nm, n = 3.5 shell): Q_sca RMS error vs cost")  # fmt: skip
    fig_shell_qsca(acc)
    out = ["# Curvilinear NFIT Mie study: summary (generated by plot_nfit.py)", "",
           "## Cost to reach a target RMS error (log-log interpolation between runs)", ""]  # fmt: skip
    t, rows = table(acc, main_arms)
    out += t + ["", "## Coated cylinder (showcase)", ""]
    t2, rows2 = table(acc, shell_arms)
    out += t2 + ["", "## All runs", ""] + runs_table(acc, main_arms + shell_arms)
    (DATA / "summary.md").write_text("\n".join(out) + "\n")
    (DATA / "targets.json").write_text(json.dumps(
        {f"{a}|{p}": v for (a, p), v in {**rows, **rows2}.items()}, indent=1))  # fmt: skip
    print("\n".join(out[:40]))


if __name__ == "__main__":
    main_plot()
