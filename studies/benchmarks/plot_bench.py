"""Figures and summary tables of the benchmark baselines (see bench_study.py)."""

import glob
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FIG = HERE / "figures"

STYLE = {
    "cart-staircase-uniform": ("tab:red", "s", ":"),
    "cart-staircase-graded": ("tab:red", "s", "-"),
    "cart-conformal-all-uniform": ("tab:orange", "D", ":"),
    "cart-conformal-all-graded": ("tab:orange", "D", "-"),
    "tri-uniform": ("tab:blue", "^", ":"),
    "tri-graded": ("tab:blue", "^", "-"),
    "cart-staircase-graded@B4": ("tab:red", "s", "-"),
    "cart-staircase-edges-graded": ("tab:blue", "^", "-"),
    "cart-conformal-pec-graded": ("tab:green", "o", "-"),
    "cart-conformal-pec-graded-bigdomain": ("tab:green", "x", "none"),
}
LABEL = {
    "cart-staircase-uniform": "Cartesian staircase, uniform",
    "cart-staircase-graded": "Cartesian staircase, graded",
    "cart-conformal-all-uniform": "Cartesian conformal, uniform",
    "cart-conformal-all-graded": "Cartesian conformal, graded",
    "tri-uniform": "triangles, uniform",
    "tri-graded": "triangles, graded",
    "cart-conformal-pec-graded": "Cartesian conformal PEC (fixed), graded",
    "cart-staircase-edges-graded": "Cartesian edge-wise staircase, graded",
    "cart-staircase-graded@B4": "Cartesian legacy STL staircase, graded",
    "cart-conformal-pec-graded-bigdomain": "conformal PEC, 2x domain",
}


def load(phase):
    recs = []
    for f in sorted(glob.glob(str(DATA / phase / "*.json"))):
        r = json.loads(Path(f).read_text())
        if "metrics" in r:
            recs.append(r)
    return recs


def by_arm(recs, case):
    out = {}
    for r in recs:
        if r["case"] == case:
            out.setdefault(r["arm"], []).append(r)
    for v in out.values():
        v.sort(key=lambda r: r["cost"]["unknowns"])
    return out


def _sim_label(arm):
    return f"Wakis FIT ({LABEL.get(arm, arm)})"


def fig_spectra(recs, cases, key, ylabel, fname, h_pick=1e-9, title=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wakis.benchmarks import get_case

    fig, axs = plt.subplots(1, len(cases), figsize=(5.2 * len(cases), 4.2),
                            constrained_layout=True, squeeze=False)  # fmt: skip
    for ax, cname in zip(axs[0], cases):
        case = get_case(cname)
        lam = case.wavelengths * 1e9
        ref = case.reference
        ax.plot(lam, ref[key], "k-", lw=2, label="exact (series / T-matrix)")
        for arm, rs in by_arm(recs, cname).items():
            r = min(rs, key=lambda r: abs(r["resolution"] - h_pick))
            if abs(r["resolution"] - h_pick) > 0.05 * h_pick:
                continue
            col, mk, _ = STYLE.get(arm, ("gray", "o", "-"))
            y = r["metrics"]["spectra"][key]
            ax.plot(lam[::3], np.asarray(y)[::3], mk, color=col, ms=4, mfc="none",
                    label=_sim_label(arm))  # fmt: skip
        ax.set_xlabel("vacuum wavelength [nm]")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{cname} (h = {h_pick * 1e9:g} nm)")
        ax.grid(alpha=0.3)
    axs[0][0].legend(fontsize=7)
    if title:
        fig.suptitle(title)
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / fname, dpi=130)
    plt.close(fig)


def fig_convergence(recs, cases, metrics, fname, xkey="unknowns"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(len(metrics), len(cases),
                            figsize=(4.6 * len(cases), 3.6 * len(metrics)),
                            constrained_layout=True, squeeze=False)  # fmt: skip
    for j, cname in enumerate(cases):
        arms = by_arm(recs, cname)
        for i, (mkey, mlabel) in enumerate(metrics):
            ax = axs[i][j]
            if mkey.endswith("lam_peak") and cname.endswith("TM"):
                mkey, mlabel = "q_abs_nrms", "Q_abs NRMS error (TM: no peak)"
            for arm, rs in arms.items():
                col, mk, ls = STYLE.get(arm, ("gray", "o", "-"))
                x = [r["cost"][xkey] if xkey != "work" else
                     r["cost"]["unknowns"] * r["cost"]["steps"] for r in rs]  # fmt: skip
                y = [r["metrics"].get(mkey, np.nan) for r in rs]
                ax.loglog(x, np.maximum(y, 1e-6), marker=mk, color=col, ls=ls,
                          mfc="none", label=_sim_label(arm))  # fmt: skip
            xs = np.array(ax.get_xlim())
            ax.loglog(
                xs,
                0.02 * (xs / xs[0]) ** -1.0,
                "k:",
                lw=0.8,
                label="slope -1 (2nd order in h)",
            )
            ax.set_xlabel({"unknowns": "field unknowns", "work": "work (unknowns x steps)",
                           "run_s": "run time [s]"}[xkey])  # fmt: skip
            ax.set_ylabel(mlabel)
            ax.set_title(cname)
            ax.grid(alpha=0.3, which="both")
    axs[0][0].legend(fontsize=6)
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / fname, dpi=130)
    plt.close(fig)


def summary_table(recs, cases, phase):
    from wakis.benchmarks import cost_to_target, get_case

    lines = [f"# {phase} baseline summary", "",
             "Cost from which each target is met and stays met on finer meshes "
             "(log-log interpolation along the h ladder; '—' = not reached; '<=' = "
             "already met by the coarsest run). Unknowns: 3 per Cartesian cell, "
             "nodes + edges for triangles. Time: time-stepping wall time on one "
             "uncontended n2-standard-2 core.", ""]  # fmt: skip
    for cname in cases:
        case = get_case(cname)
        arms = by_arm(recs, cname)
        if not arms:
            continue
        lines += [f"## {cname}", "",
                  "| metric | target | arm | unknowns | work | time [s] |",
                  "|---|---|---|---|---|---|"]  # fmt: skip
        for metric, tgts in case.targets.items():
            for t in tgts:
                for arm, rs in sorted(arms.items()):
                    c = cost_to_target(rs, metric, t)
                    fmt = lambda v, f: (
                        "—"
                        if not np.isfinite(v)
                        # noqa: E731
                        else ("<= " if v < 0 else "") + f.format(abs(v))
                    )
                    lines.append(
                        f"| {metric} | {t:g} | {arm} | {fmt(c['unknowns'], '{:.3g}')} | "
                        f"{fmt(c['work'], '{:.3g}')} | {fmt(c['run_s'], '{:.0f}')} |"
                    )
        lines += ["", "Per-run records:", "",
                  "| arm | h [nm] | unknowns | steps | run [s] | Q_ext NRMS | lambda_peak err | "
                  "height err | FWHM err | enh peak err | reflection |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
        for arm, rs in sorted(arms.items()):
            for r in rs:
                m, c = r["metrics"], r["cost"]
                g = (
                    lambda k: f"{m[k]:.2e}"
                    if isinstance(m.get(k), float) and np.isfinite(m[k])
                    else "—"
                )  # noqa: E731
                lines.append(
                    f"| {arm} | {r['resolution'] * 1e9:.3g} | {c['unknowns']} | {c['steps']} | "
                    f"{c['run_s']:.0f} | {g('q_ext_nrms')} | {g('q_ext_lam_peak')} | "
                    f"{g('q_ext_height')} | {g('q_ext_fwhm')} | {g('enh_peak')} | "
                    f"{g('boundary_reflection')} |"
                )
        lines.append("")
    (DATA / f"summary_{phase}.md").write_text("\n".join(lines))


def fig_b3_reference_map(fname="b3_reference_field_map.png"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.constants import c as c0

    from wakis.benchmarks import get_case
    from wakis.benchmarks import metrics as M
    from wakis.benchmarks.meshing import cluster_mesh

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    for ax, g in zip(axs[:2], (5, 2)):
        case = get_case(f"B3-g{g}")
        p = M.peak(case.wavelengths, case.reference["enh"])
        cl = case.cluster(c0 / p["lam_peak"])
        x = np.linspace(-70e-9, 70e-9, 141)
        y = np.linspace(-45e-9, 45e-9, 91)
        X, Y = np.meshgrid(x, y)
        pts = np.stack([X.ravel(), Y.ravel()], 1)
        inside = np.zeros(len(pts), bool)
        for cy in case.cylinders:
            inside |= (
                np.hypot(*(pts - np.asarray(cy["center"])).T) < cy["radius"] * 1.001
            )
        E = np.full(len(pts), np.nan)
        E[~inside] = np.sum(np.abs(cl.in_plane_e(pts[~inside])) ** 2, 1)
        im = ax.pcolormesh(X * 1e9, Y * 1e9, np.log10(E.reshape(X.shape)), cmap="magma",
                           shading="auto")  # fmt: skip
        for cy in case.cylinders:
            ax.add_patch(plt.Circle(np.asarray(cy["center"]) * 1e9, cy["radius"] * 1e9,
                                    fill=False, color="w", lw=0.8))  # fmt: skip
        ax.set_aspect("equal")
        ax.set_title(
            f"exact log10|E/E0|^2, g = {g} nm, lambda = {p['lam_peak'] * 1e9:.0f} nm"
        )
        ax.set_xlabel("x [nm]")
        ax.set_ylabel("y [nm]")
        plt.colorbar(im, ax=ax, shrink=0.8)
    case = get_case("B3-g5")
    mesh = cluster_mesh(case.cylinders, 2e-9, 8e-9, 110e-9, pml_layers=4, pml_dr=8e-9)
    P = mesh.nodes[mesh.cells]
    from matplotlib.collections import PolyCollection

    cols = np.where(mesh.cell_tags[:, None] == 0, [[0.9, 0.9, 0.95]],
                    np.where(mesh.cell_tags[:, None] == 3, [[0.8, 0.95, 0.8]], [[1.0, 0.8, 0.3]]))  # fmt: skip
    axs[2].add_collection(
        PolyCollection(P * 1e9, facecolors=cols, edgecolors="k", lw=0.15)
    )
    axs[2].set_xlim(-130, 130)
    axs[2].set_ylim(-130, 130)
    axs[2].set_aspect("equal")
    axs[2].set_title(f"tri-graded, h = 2 nm, g = 5 nm ({mesh.n_nodes} nodes)")
    axs[2].set_xlabel("x [nm]")
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / fname, dpi=130)
    plt.close(fig)


def fig_b4(recs, fname="b4_dipole_zin.png"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wakis.benchmarks import get_case

    case = get_case("B4")
    ref = case.reference
    f = case.freqs / 1e9
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    axs[0].plot(f, ref["z_in"].real, "k-", lw=2, label="R, Hallen MoM (reference)")
    axs[0].plot(f, ref["z_in"].imag, "k--", lw=2, label="X, Hallen MoM")
    arms = {k: v for k, v in by_arm(recs, "B4").items()
            if k in ("cart-staircase-edges-graded", "cart-staircase-graded",
                     "cart-conformal-pec-graded")}  # fmt: skip
    for arm, rs in arms.items():
        col, mk, _ = STYLE.get(arm, ("gray", "o", "-"))
        r = rs[-1]  # finest
        sp = r["metrics"]["spectra"]
        axs[0].plot(f[::2], np.asarray(sp["z_re"])[::2], mk, color=col, mfc="none", ms=4,
                    label=f"{_sim_label(arm)} h_t = {r['resolution'] * 1e3:.3g} mm")  # fmt: skip
        axs[0].plot(f[::2], np.asarray(sp["z_im"])[::2], mk, color=col, ms=3)
        h = [r["resolution"] * 1e3 for r in rs]
        fr = [r["metrics"].get("f_res", np.nan) for r in rs]
        rr = [r["metrics"].get("r_res", np.nan) for r in rs]
        axs[1].loglog(
            h, fr, marker=mk, color=col, label=f"{LABEL.get(arm, arm)}: f_res"
        )
        axs[1].loglog(h, rr, marker=mk, color=col, ls="--", mfc="none",
                      label=f"{LABEL.get(arm, arm)}: R_res")  # fmt: skip
    axs[0].set_xlabel("frequency [GHz]")
    axs[0].set_ylabel("Z_in [Ohm]")
    axs[0].grid(alpha=0.3)
    axs[0].legend(fontsize=7)
    axs[1].set_xlabel("transverse cell at the wire h_t [mm]")
    axs[1].set_ylabel("relative error")
    axs[1].grid(alpha=0.3, which="both")
    axs[1].legend(fontsize=7)
    axs[1].set_title(
        f"B4: reference f_res = {ref['f_res'] / 1e9:.4f} GHz, R = {ref['r_res']:.1f} Ohm"
    )
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / fname, dpi=130)
    plt.close(fig)


def main_plot(phase):
    recs = load(phase)
    if phase == "B2":
        fig_spectra(
            recs, ["B2-Au-TE", "B2-Ag-TE"], "q_ext", "Q_ext", "b2_qext_spectra.png"
        )
        fig_spectra(
            recs, ["B2-Au-TE", "B2-Ag-TE"], "q_abs", "Q_abs", "b2_qabs_spectra.png"
        )
        fig_convergence(recs, ["B2-Au-TE", "B2-Ag-TE", "B2-Au-TM"],
                        [("q_ext_nrms", "Q_ext NRMS error"),
                         ("q_ext_lam_peak", "resonance wavelength error")],
                        "b2_convergence.png")  # fmt: skip
        fig_convergence(recs, ["B2-Au-TE", "B2-Ag-TE"], [("q_ext_nrms", "Q_ext NRMS error")],
                        "b2_error_vs_time.png", xkey="run_s")  # fmt: skip
        summary_table(recs, ["B2-Au-TE", "B2-Au-TM", "B2-Ag-TE", "B2-Ag-TM"], "B2")
    elif phase in ("B3", "B3m2"):
        sfx = "_m2" if phase == "B3m2" else ""
        fig_spectra(recs, ["B3-g10", "B3-g5", "B3-g2"], "enh", "gap |E/E0|^2",
                    f"b3_gap_enhancement_spectra{sfx}.png")  # fmt: skip
        fig_spectra(recs, ["B3-g10", "B3-g5", "B3-g2"], "q_ext", "Q_ext (per 4a)",
                    f"b3_qext_spectra{sfx}.png")  # fmt: skip
        fig_convergence(recs, ["B3-g10", "B3-g5", "B3-g2"],
                        [("enh_peak", "peak gap enhancement error"),
                         ("q_ext_nrms", "Q_ext NRMS error")],
                        f"b3_convergence{sfx}.png")  # fmt: skip
        if not sfx:
            fig_b3_reference_map()
        summary_table(recs, ["B3-g10", "B3-g5", "B3-g2"], phase)
    elif phase == "B4":
        fig_b4(recs)
        summary_table(recs, ["B4"], "B4")
