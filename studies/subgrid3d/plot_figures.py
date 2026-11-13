"""
Figures and summary numbers of the 3D subgridding study from results.json.

Writes docs/img/subgrid3d_reflection.png, docs/img/subgrid3d_cavity_convergence.png,
docs/img/subgrid3d_cost.png and studies/subgrid3d/summary.json.
Usage: python studies/subgrid3d/plot_figures.py [results.json]
"""

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(HERE, "..", "..", "docs", "img")
COL = {0: "#0072B2", 15: "#56B4E9", 30: "#009E73", 45: "#E69F00", 60: "#D55E00", 75: "#CC79A7"}
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3, "savefig.dpi": 200})


def order_guide(ax, x, y0, p, label):
    x = np.asarray(x, float)
    ax.loglog(x, y0 * (x / x[0]) ** (-p), "k:", lw=1)
    ax.text(x[-1], y0 * (x[-1] / x[0]) ** (-p), f"  {label}", va="center", fontsize=8)


def fig_reflection(res, summ):
    R = res["reflection"]
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.4))
    for ax, (ratio, pol) in zip(axs[:2], ((2, "s"), (2, "p"))):
        for th in sorted({r["theta"] for r in R}):
            pts = sorted((r["N"], r["R"]) for r in R
                         if r["kind"] == "3d" and r["ratio"] == ratio and r["pol"] == pol and r["theta"] == th)  # fmt: skip
            if pts:
                N, v = zip(*pts)
                ax.loglog(N, v, "o-", ms=3.5, color=COL.get(th, "k"), label=f"{th}°")
            pts3 = sorted((r["N"], r["R"]) for r in R
                          if r["kind"] == "3d" and r["ratio"] == 3 and r["pol"] == pol and r["theta"] == th)  # fmt: skip
            if pts3 and th in (0, 60):
                N, v = zip(*pts3)
                ax.loglog(N, v, "s--", ms=3, mfc="none", color=COL.get(th, "k"), lw=0.8,
                          label=f"{th}° (3:1)")  # fmt: skip
        fl = sorted((r["N"], r["R"]) for r in R if r["kind"] == "floor" and r["pol"] == pol and r["theta"] == 30)
        if fl:
            N, v = zip(*fl)
            ax.loglog(N, v, "x", color="0.5", label="floor (no patch)")
        r1 = sorted((r["N"], r["R1d"]) for r in R if r["kind"] == "3d" and r["theta"] == 0 and r["ratio"] == 2
                    and r["pol"] == pol)  # fmt: skip
        if r1:
            N, v = zip(*r1)
            ax.loglog(N, v, "k+", ms=8, label="1D exact (0°)")
            order_guide(ax, N, 1.6 * v[0], 2, "$N^{-2}$")
        ax.set_xlabel("coarse cells per wavelength $N$")
        ax.set_ylabel("$|R|$")
        ax.set_title(f"({'a' if pol == 's' else 'b'}) {pol}-pol., 2:1 (solid), 3:1 (dashed)")
        ax.legend(fontsize=6.5, ncol=2)
    ax = axs[2]
    for N, mk in ((10, "o"), (20, "s")):
        for pol, ls in (("s", "-"), ("p", "--")):
            pts = sorted((r["theta"], r["R"]) for r in R
                         if r["kind"] == "3d" and r["ratio"] == 2 and r["pol"] == pol and r["N"] == N)  # fmt: skip
            if pts:
                t, v = zip(*pts)
                ax.semilogy(t, v, mk + ls, ms=3.5, label=f"3D {pol}, N={N}")
        pts = sorted((r["theta"], r["R"]) for r in R if r["kind"] == "2d" and r["ratio"] == 2 and r["N"] == N)
        if pts:
            t, v = zip(*pts)
            ax.semilogy(t, v, "k" + mk, mfc="none", ms=7, label=f"2D TM, N={N}")
    ax.set_xlabel("angle of incidence θ [deg]")
    ax.set_ylabel("$|R|$")
    ax.set_title("(c) 2:1 interface vs angle")
    ax.legend(fontsize=6.5, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "subgrid3d_reflection.png"))
    plt.close(fig)
    # summary numbers
    out = {}
    for r in R:
        if r["kind"] == "3d":
            out[f"R_{r['pol']}_r{r['ratio']}_th{r['theta']}_N{r['N']}"] = r["R"]
    orders = {}
    for ratio in (2, 3):
        for pol in ("s", "p"):
            for th in sorted({r["theta"] for r in R}):
                pts = sorted((r["N"], r["R"]) for r in R
                             if r["kind"] == "3d" and r["ratio"] == ratio and r["pol"] == pol and r["theta"] == th)  # fmt: skip
                if len(pts) >= 2:
                    (n0, v0), (n1, v1) = pts[-2], pts[-1]
                    orders[f"{pol}_r{ratio}_th{th}"] = float(np.log(v0 / v1) / np.log(n1 / n0))
    eq = [abs(a["R"] / b["R"] - 1) for a in R if a["kind"] == "3d_eqsrc" for b in R
          if b["kind"] == "2d" and b["N"] == a["N"] and b["theta"] == a["theta"] and b["ratio"] == a["ratio"]]  # fmt: skip
    d1 = [abs(r["R"] - r["R1d"]) for r in R if r["kind"] == "3d" and r["theta"] == 0]
    fl = [r["R"] for r in R if r["kind"] == "floor"]
    summ["reflection"] = {"values": out, "orders_last_pair": orders,
                          "max_rel_diff_3d_s_vs_2d_tm": max(eq) if eq else None,
                          "max_abs_diff_normal_vs_1d_exact": max(d1) if d1 else None,
                          "max_floor": max(fl) if fl else None,
                          "mutations_th60_s_r2": {k: sorted((r["N"], r["R"]) for r in R if r["kind"] == k)
                                                  for k in ("fraction", "missing_transverse", "inject")}}  # fmt: skip


def fig_cavity(res, summ):
    C = res["cavity"]
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    style = {"uniform": ("k", "o", "uniform coarse (= SolverFIT3D)"), "r2": ("#0072B2", "s", "2:1 patch"),
             "r3": ("#009E73", "^", "3:1 patch"), "nested": ("#E69F00", "D", "nested 2:1 + 2:1"),
             "fraction": ("#D55E00", "x", "mutation: wrong $P_e$ fraction")}  # fmt: skip
    orders = {}
    for arm, (c, mk, lab) in style.items():
        pts = sorted((r["h"], max(r["err"])) for r in C if r["arm"] == arm)
        if not pts:
            continue
        h, e = zip(*pts)
        ax.loglog(h, e, mk + "-", color=c, ms=4, label=lab)
        if len(pts) >= 2:
            p = np.polyfit(np.log(h), np.log(e), 1)[0]
            orders[arm] = {"fit_all": float(p),
                           "last_pair": float(np.log(e[1] / e[0]) / np.log(h[1] / h[0]))}  # fmt: skip
    h = sorted({r["h"] for r in C})
    e0 = max(max(r["err"]) for r in C if r["arm"] == "uniform" and r["h"] == h[-1])
    ax.loglog(h, 1.4 * e0 * (np.array(h) / h[-1]) ** 2, "k:", lw=1)
    ax.text(h[0], 1.4 * e0 * (h[0] / h[-1]) ** 2, "$h^2$  ", ha="right", va="center")
    ax.set_xlabel("coarse cell size $h$ [m]")
    ax.set_ylabel("max rel. error, 6 lowest modes")
    ax.set_title("PEC box 1 × 0.75 × 0.5 m, off-centre patch")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "subgrid3d_cavity_convergence.png"))
    plt.close(fig)
    summ["cavity"] = {"orders": orders,
                      "errors": {f"{r['arm']}_N{r['N']}": max(r["err"]) for r in C},
                      "unknowns": {f"{r['arm']}_N{r['N']}": r["n_free"] + r["n_faces"] for r in C}}  # fmt: skip


def fig_cost(res, summ):
    L = res["loaded"]
    uni = sorted((r for r in L if r["ratio"] == 1), key=lambda r: -r["h_fine"])
    hs = np.array([r["h_fine"] for r in uni])
    fs = np.array([r["f_semi_probes"][0] for r in uni])
    # Richardson: f = f* + a h^2 + b h^4 on the three finest uniform arms
    A2 = np.stack([np.ones(2), hs[-2:] ** 2], 1)
    fstar2 = float(np.linalg.solve(A2, fs[-2:])[0])
    if len(hs) >= 3:
        A = np.stack([np.ones(3), hs[-3:] ** 2, hs[-3:] ** 4], 1)
        fstar = float(np.linalg.solve(A, fs[-3:])[0])
    else:  # --quick
        fstar = fstar2
    err = lambda r: abs(r["f_semi_probes"][0] / fstar - 1)  # noqa: E731
    T = {round(t["h"], 6): t for t in res.get("timing", [])}
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
    arms = {1: ("k", "o", "uniform $h$ (SolverFIT3D discretisation)"),
            3: ("#009E73", "^", "subgrid, $H = 3h$, 3:1 patch"),
            2: ("#0072B2", "s", "subgrid, $H = 2h$, 2:1 patch")}  # fmt: skip
    table = []
    for ratio, (c, mk, lab) in arms.items():
        rs = sorted((r for r in L if r["ratio"] == ratio), key=lambda r: r["unknowns"])
        if not rs:
            continue
        u = [r["unknowns"] for r in rs]
        w = [r["unknowns"] * r["n_steps"] for r in rs]
        e = [err(r) for r in rs]
        axs[0].loglog(u, e, mk + "-", color=c, ms=4, label=lab)
        axs[1].loglog(w, e, mk + "-", color=c, ms=4, label=lab)
        for r in rs:
            table.append({"ratio": ratio, "h_coarse": r["h_coarse"], "h_fine": r["h_fine"],
                          "unknowns": r["unknowns"], "n_steps": r["n_steps"], "dt": r["dt"],
                          "work": r["unknowns"] * r["n_steps"], "err": err(r),
                          "run_s": r["run_s"], "s_per_step": r["s_per_step"]})  # fmt: skip
    for ax, xl in zip(axs, ("unknowns (edges + faces)", "work: unknowns × steps (same physical time)")):
        ax.set_xlabel(xl)
        ax.set_ylabel("rel. error of $f_1$ (semi-discrete)")
        ax.legend(fontsize=7)
    axs[0].set_title("(a) PEC box with $\\varepsilon_r = 9$ cube: accuracy vs size")
    axs[1].set_title("(b) accuracy vs work")
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "subgrid3d_cost.png"))
    plt.close(fig)

    def interp(xs, es, e_target):
        """log-log interpolation of x at e_target along a monotone curve."""
        xs, es = np.log(np.asarray(xs, float)), np.log(np.asarray(es, float))
        o = np.argsort(es)
        if not (es.min() <= np.log(e_target) <= es.max()):
            return None
        return float(np.exp(np.interp(np.log(e_target), es[o], xs[o])))

    ue = [err(r) for r in uni]
    eq = []
    for t in table:
        if t["ratio"] == 1:
            continue
        uu = interp([r["unknowns"] for r in uni], ue, t["err"])
        uw = interp([r["unknowns"] * r["n_steps"] for r in uni], ue, t["err"])
        eq.append({**t, "uniform_unknowns_at_equal_err": uu, "uniform_work_at_equal_err": uw,
                   "unknown_ratio": None if uu is None else uu / t["unknowns"],
                   "work_ratio": None if uw is None else uw / t["work"]})  # fmt: skip
    summ["cost"] = {"f_star_richardson_h2h4": fstar, "f_star_richardson_h2_last2": fstar2,
                    "uniform_orders": [float(np.log(abs(fs[i] - fstar) / abs(fs[i + 1] - fstar)) /
                                             np.log(hs[i] / hs[i + 1])) for i in range(len(hs) - 1)],
                    "arms": table, "equal_accuracy": eq, "timing": list(T.values()),
                    "guess": res.get("loaded_guess")}  # fmt: skip


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results.json")
    with open(path) as f:
        res = json.load(f)
    os.makedirs(IMG, exist_ok=True)
    summ = {}
    if "reflection" in res:
        fig_reflection(res, summ)
    if "cavity" in res:
        fig_cavity(res, summ)
    if "loaded" in res:
        fig_cost(res, summ)
    with open(os.path.join(HERE, "summary.json"), "w") as f:
        json.dump(summ, f, indent=1)
    print(json.dumps({k: v for k, v in summ.items() if k != "reflection"}, indent=1)[:4000])


if __name__ == "__main__":
    main()
