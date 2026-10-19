"""Figures and summary table of the LTS study (see README)."""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA, FIG = HERE / "data", HERE / "figures"
COL = {"global": "0.35", "lts": "C3"}
LS = {"tri-graded": "-", "tri-graded-compact": "--"}


def rms(d):
    return float(np.sqrt(np.mean((np.array(d["q"]) / np.array(d["q_ref"]) - 1) ** 2)))


def load():
    runs = {}
    for f in DATA.glob("tri-*.json"):
        d = json.loads(f.read_text())
        if "error" in d:
            print("ERROR", f.name, d["error"])
            continue
        runs[(d["arm"], d["pol"], d["res"], d["mode"])] = d
    return runs


def fig_partition():
    sys.path.insert(0, str(HERE))
    import lts_study as LS_

    mesh, s, h = LS_.build("tri-graded", "TM", 10, "lts")
    g, _, _ = LS_.build("tri-graded", "TM", 10, "global")[1], None, None
    L = s.lts
    cls = np.zeros(mesh.n_nodes, int)  # 0 fixed/coarse
    cls[s.free[L.C]] = 1
    cls[s.free[L.F]] = 2
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.8), gridspec_kw=dict(width_ratios=[1.1, 1]))
    xy = mesh.nodes * 1e6
    tri = mesh.cells
    q = (xy[tri].mean(1) > -0.05).all(1)
    ax[0].triplot(xy[:, 0], xy[:, 1], tri[q], lw=0.25, color="0.6")
    names = ["coarse (dt)", "buffer: coarse, coupled to F", "fine F (2 substeps of dt/2)"]
    cols = ["0.75", "C1", "C0"]
    for k in range(3):
        sel = (cls == k) & (xy > -0.05).all(1)
        ax[0].plot(*xy[sel].T, ".", ms=2.5 if k else 1.2, color=cols[k], label=names[k])
    th = np.linspace(0, np.pi / 2, 200)
    ax[0].plot(np.cos(th), np.sin(th), "k-", lw=1)
    ax[0].set_aspect("equal")
    ax[0].set_xlim(0, 2.75)
    ax[0].set_ylim(0, 2.75)
    ax[0].set_xlabel("x [µm]")
    ax[0].set_ylabel("y [µm]")
    ax[0].legend(loc="upper right", fontsize=8, markerscale=4)
    ax[0].set_title(
        f"(a) LTS levels, graded Mie mesh res 10: {L.F.size / L.n:.1%} fine, "
        f"dt = {s.dt / g.dt:.2f}× global",
        fontsize=9,
    )
    # local CFL limit vs radius
    from wakis.unstructured.lts import gershgorin_rate

    rate = gershgorin_rate(g.K, g.mass)
    loc = 1 / np.sqrt(rate / rate.max())  # local limit / global limit (estimate)
    r = np.hypot(*mesh.nodes[g.free].T) * 1e6
    inF = np.zeros(g.free.size, bool)
    inF[L.F] = True
    ax[1].plot(r[~inF], loc[~inF], ".", ms=1.5, color="0.6", label="coarse")
    ax[1].plot(r[inF], loc[inF], ".", ms=2, color="C0", label="fine F")
    ax[1].axhline(s.dt / g.dt, color="C3", lw=1, label="LTS coarse dt / global dt")
    ax[1].axvline(1.0, color="k", lw=0.8, ls=":")
    ax[1].text(0.5, 2.55, "cylinder\nn = 2", ha="center", fontsize=8)
    ax[1].text(1.6, 2.55, "vacuum, h grows h_in → 2 h_in", ha="center", fontsize=8)
    ax[1].set_xlabel("node radius [µm]")
    ax[1].set_ylabel("local step limit / global limit (Gershgorin)")
    ax[1].set_ylim(0.8, 2.8)
    ax[1].legend(fontsize=8, loc="lower right", markerscale=4)
    ax[1].set_title("(b) node-local CFL estimate: only the vacuum band is fine", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "a_partition.png", dpi=160)


def fig_energy():
    d = json.loads((DATA / "energy.json").read_text())
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    style = {
        "global": ("0.35", "global leapfrog (dt)"),
        "LTS nu=0.15": ("C3", "LTS p=2, ν=0.15 (2 dt)"),
        "nu=0": ("C2", "LTS ν=0 (Diaz–Grote, unstabilized)"),
        "mutant: buffer": ("C1", "mutant: F decoupled from buffer"),
        "mutant: normalization": ("C4", "mutant: r(0)=1.01"),
    }
    for k, (c, lab) in style.items():
        v = d[k]
        t = np.array(v["t"]) * 1e6
        E = np.array(v["E"])
        ax[0].semilogy(t, np.abs(E / E[0] - 1) + 1e-17, color=c, lw=1, label=lab)
        if k in ("LTS nu=0.15", "nu=0", "global"):
            P = np.array(v["P"])
            ax[1].semilogy(t, P / P[0], color=c, lw=1, label=lab)
    ax[0].set_xlabel("time [µs] (10⁵ coarse steps)")
    ax[0].set_ylabel("|E/E₀ − 1| (exact LTS energy)")
    ax[0].set_title("(a) conserved LTS energy, PEC cavity, TM", fontsize=9)
    ax[0].legend(fontsize=7)
    ax[1].set_xlabel("time [µs]")
    ax[1].set_ylabel("physical energy ½uᵀMu + ½vᵀSv, relative")
    ax[1].set_title("(b) ν = 0 is marginal (ρ = 1): field energy beats up to ~300×", fontsize=9)
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "b_energy.png", dpi=160)


def fig_speedup(runs):
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    for pol, mk in (("TM", "o"), ("TE", "s")):
        for arm in LS:
            for mode in ("global", "lts"):
                ks = sorted(k for k in runs if k[:2] == (arm, pol) and k[3] == mode)
                if not ks:
                    continue
                dofs = [runs[k]["dofs"] for k in ks]
                err = [rms(runs[k]) for k in ks]
                tt = [runs[k]["run_s"] for k in ks]
                lab = f"{pol} {arm} {mode}"
                ax[0].loglog(dofs, err, LS[arm], marker=mk, color=COL[mode], ms=4, lw=1, label=lab)
                ax[1].loglog(tt, err, LS[arm], marker=mk, color=COL[mode], ms=4, lw=1, label=lab)
            ks = sorted(k for k in runs if k[:2] == (arm, pol) and k[3] == "lts")
            sp, sps, res = [], [], []
            for k in ks:
                g = runs.get((arm, pol, k[2], "global"))
                if g:
                    res.append(k[2])
                    sp.append(g["run_s"] / runs[k]["run_s"])
                    sps.append(g["step_s"] / runs[k]["step_s"])
            ax[2].plot(res, sp, LS[arm], marker=mk, color="C3", lw=1, label=f"{pol} {arm}: run")
            ax[2].plot(res, sps, LS[arm], marker=mk, color="C0", lw=1, mfc="none",
                       label=f"{pol} {arm}: field update only")  # fmt: skip
    ax[0].set_xlabel("field unknowns (nodes + edges)")
    ax[0].set_ylabel("RMS relative error of Q_sca (x = 2–10)")
    ax[0].set_title("(a) accuracy: LTS (red) on top of global (grey)", fontsize=9)
    ax[1].set_xlabel("uncontended run time [s] (n2-standard-2, 1 core)")
    ax[1].set_title("(b) error vs wall time", fontsize=9)
    ax[0].legend(fontsize=6, ncol=2)
    ax[2].axhline(1, color="k", lw=0.6)
    ax[2].set_xlabel("res (cells per λ in the dielectric)")
    ax[2].set_ylabel("speed-up (global time / LTS time)")
    ax[2].set_title("(c) LTS speed-up at the same mesh", fontsize=9)
    ax[2].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(FIG / "c_speedup.png", dpi=160)


def fine_edges(runs):
    """Edges whose flux gets the LTS correction (rows of G[:, F]); cached."""
    f = DATA / "fine_edges.json"
    cache = json.loads(f.read_text()) if f.exists() else {}
    sys.path.insert(0, str(HERE))
    import lts_study as LS_

    for k in runs:
        key = "_".join(map(str, k))
        if k[3] == "lts" and key not in cache:
            _, s, _ = LS_.build(*k)
            cache[key] = int((abs(s._G_F).sum(axis=1) > 0).sum())
    f.write_text(json.dumps(cache))
    return cache


def work(d, ef):
    """Unknown-weighted work: unknown updates over the run. LTS adds, per
    coarse step and per extra substep, the fine nodes and their edges."""
    if d["mode"] == "global":
        return d["dofs"] * d["steps"]
    return d["steps"] * (d["dofs"] + d["fine"] + ef)


def summary(runs):
    ef = fine_edges(runs)
    lines = [
        "| pol | arm | res | dofs | global: dt [as] / steps / RMS / run [s] (update) "
        "| LTS: dt / steps / RMS / run [s] (update) | fine % | rho "
        "| work ratio (unknown-weighted) | speed-up run (update) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for pol in ("TM", "TE"):
        for arm in LS:
            for res in (10, 14, 20, 28, 40):
                g, s = runs.get((arm, pol, res, "global")), runs.get((arm, pol, res, "lts"))
                if not (g and s):
                    continue
                lines.append(
                    f"| {pol} | {arm} | {res} | {g['dofs']} "
                    f"| {g['dt'] * 1e18:.1f} / {g['steps']} / {rms(g):.4f} / {g['run_s']:.0f} ({g['step_s']:.0f}) "
                    f"| {s['dt'] * 1e18:.1f} / {s['steps']} / {rms(s):.4f} / {s['run_s']:.0f} ({s['step_s']:.0f}) "
                    f"| {100 * s['fine'] / s['nodes']:.1f} | {s['rho']:.3f} "
                    f"| {work(g, 0) / work(s, ef['_'.join(map(str, (arm, pol, res, 'lts')))]):.2f} "
                    f"| {g['run_s'] / s['run_s']:.2f} ({g['step_s'] / s['step_s']:.2f}) |"
                )
    (DATA / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main_plot():
    FIG.mkdir(exist_ok=True)
    runs = load()
    fig_partition()
    fig_energy()
    if runs:
        fig_speedup(runs)
        summary(runs)


if __name__ == "__main__":
    main_plot()
