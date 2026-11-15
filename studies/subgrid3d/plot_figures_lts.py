"""
Figures and summary of the wave-2 study (`run_study_lts.py`).

Reads results_D.json, results_E.json, results_F.json, results_G.json (any
subset) and writes docs/img/subgrid3d_lts_*.png and summary_lts.json.

Accuracy: relative error of the lowest resonance against f*, the 3-point
power-law extrapolation of the three finest uniform arms (semi-discrete
frequencies, as in wave 1). Two error measures per arm:
  err_semi - headline spatial error: f_td mapped through the leapfrog
             dispersion of the arm's own (coarse) step,
             2 sin(pi f dt)/(pi dt)/2, removing the leapfrog time error;
  err_td   - secondary time-domain error (time error included).
Equal accuracy: for each subgrid or graded arm, the uniform and wave-1
global-dt subgrid work and wall clock at the same error by log-log
interpolation along the reference curve.
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(HERE, "..", "..", "docs", "img")


def load(part):
    p = os.path.join(HERE, f"results_{part}.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def power_law_ref(h, f):
    """f(h) = f* + a h^p through the three finest points."""
    i = np.argsort(h)[:3]
    h, f = np.asarray(h, float)[i], np.asarray(f, float)[i]
    if len(h) < 3:
        p = 2.0
        a = (f[1] - f[0]) / (h[1] ** p - h[0] ** p)
        return float(f[0] - a * h[0] ** p), float(p)
    # h ratios may differ: solve for p by bisection on the ratio equation
    from scipy.optimize import brentq

    def g(p):
        return (f[1] - f[0]) / (f[2] - f[1]) - (h[1] ** p - h[0] ** p) / (
            h[2] ** p - h[1] ** p
        )

    if f[2] != f[1] and g(0.3) * g(4.0) <= 0:
        p = brentq(g, 0.3, 4.0)
        a = (f[1] - f[0]) / (h[1] ** p - h[0] ** p)
        return float(f[0] - a * h[0] ** p), float(p)
    p = 2.0
    a, f0 = np.polyfit(h**p, f, 1)
    return float(f0), float(p)


def interp_loglog(x_err, err_curve, val_curve):
    """val at error x_err along the curve (log-log, linear; extrapolates)."""
    if len(err_curve) == 1:
        return float(val_curve[0]), False
    o = np.argsort(err_curve)
    le, lv = np.log(np.asarray(err_curve)[o]), np.log(np.asarray(val_curve)[o])
    x = np.log(x_err)
    if x <= le[0]:
        j = 0
    elif x >= le[-1]:
        j = len(le) - 2
    else:
        j = int(np.searchsorted(le, x) - 1)
    t = (x - le[j]) / (le[j + 1] - le[j])
    inside = le[0] <= x <= le[-1]
    return float(np.exp(lv[j] + t * (lv[j + 1] - lv[j]))), bool(inside)


def _track_modes(rows, f_seed=None):
    """Track one mode per probe by continuity across h (finest to coarsest)."""
    groups = {}
    for r in rows:
        key = (
            r["kind"] if r["kind"] in ("uniform", "graded") else (r["kind"], r["ratio"])
        )
        groups.setdefault(key, []).append(r)
    for key in sorted(groups, key=lambda k: (0 if k == "uniform" else 1, str(k))):
        g = sorted(groups[key], key=lambda r: r["h_fine"])
        f_prev = None
        for r in g:
            c_semi = r.get("modes_semi_probes") or [[v] for v in r["f_semi_probes"]]
            c_td = r.get("modes_td_probes") or [[v] for v in r["f_td_probes"]]
            if f_prev is None:
                seed = (
                    f_seed
                    if f_seed is not None
                    else float(np.mean([c[0] for c in c_semi]))
                )
                f_prev = [seed] * len(c_semi)
            tracked_semi, tracked_td = [], []
            for p, (cs, ct) in enumerate(zip(c_semi, c_td)):
                idx = int(np.argmin(np.abs(np.asarray(cs, float) - f_prev[p])))
                tracked_semi.append(float(cs[idx]))
                tracked_td.append(float(ct[idx]))
            r["f_semi_tracked"] = tracked_semi
            r["f_td_tracked"] = tracked_td
            f_prev = tracked_semi


def analyse_cost(res):
    rows = res["cost"]
    tim = {
        (t["kind"], round(t["h_coarse"], 6), t["ratio"]): t
        for t in res.get("timing", [])
    }
    _track_modes(rows, f_seed=res.get("f1_guess"))
    uni = [r for r in rows if r["kind"] == "uniform"]
    fstar, order = power_law_ref([r["h_fine"] for r in uni],
                                 [float(np.mean(r["f_semi_tracked"])) for r in uni])  # fmt: skip
    for r in rows:
        f_semi = r["f_semi_tracked"]
        f_td = r["f_td_tracked"]
        r["err_semi"] = float(abs(np.mean(f_semi) / fstar - 1))
        r["err_td"] = float(abs(np.mean(f_td) / fstar - 1))
        r["probe_diff_semi"] = float(abs(f_semi[0] - f_semi[1]) / fstar)
        r["probe_agreement_ratio"] = float(
            r["probe_diff_semi"] / max(r["err_semi"], 1e-30)
        )
        t = tim.get((r["kind"], round(r["h_coarse"], 6), r["ratio"]))
        r["s_per_step"] = t["s_per_step"] if t else None
        r["wall"] = r["s_per_step"] * r["n_steps"] if t else None
        if t and "solverfit3d_s_per_step" in t:
            r["wall_solverfit3d"] = t["solverfit3d_s_per_step"] * r["n_steps"]
    uni = sorted(
        [r for r in rows if r["kind"] == "uniform"], key=lambda r: -r["h_fine"]
    )
    glb_all = [r for r in rows if r["kind"] == "global"]
    by_key = {(r["kind"], round(r["h_coarse"], 6), r["ratio"]): r for r in rows}
    max_ratio = float(max(r["probe_agreement_ratio"] for r in rows))
    out = {"f_star": fstar, "observed_order": order,
           "max_probe_agreement_ratio": max_ratio, "arms": []}  # fmt: skip
    for r in rows:
        if r["kind"] == "uniform":
            continue
        a = {k: r[k] for k in ("kind", "h_coarse", "ratio", "h_fine", "unknowns", "dt",
                               "dt_global_max", "n_steps", "work", "err_semi", "err_td",
                               "probe_agreement_ratio", "s_per_step", "wall")}  # fmt: skip
        if "wall_solverfit3d" in r:
            a["wall_solverfit3d"] = r["wall_solverfit3d"]
        if r["kind"] == "lts":
            a["dt_lts_over_global"] = r["dt"] / r["dt_global_max"]
            a["rho"] = r.get("rho")
            a["F1"] = r["lts"].get("F1")
        glb_r = [g for g in glb_all if g["ratio"] == r["ratio"]] or glb_all
        g_same = by_key.get(("global", round(r["h_coarse"], 6), r["ratio"]))
        for m in ("semi", "td"):
            w, ins = interp_loglog(r[f"err_{m}"], [u[f"err_{m}"] for u in uni],
                                   [u["work"] for u in uni])  # fmt: skip
            a[f"work_gain_{m}"] = w / r["work"]
            a[f"in_range_{m}"] = ins
            if r["wall"] and all(u["wall"] for u in uni):
                w, _ = interp_loglog(r[f"err_{m}"], [u[f"err_{m}"] for u in uni],
                                     [u["wall"] for u in uni])  # fmt: skip
                a[f"wall_gain_{m}"] = w / r["wall"]
            if len(glb_r) >= 2:
                wg, _ = interp_loglog(r[f"err_{m}"], [g[f"err_{m}"] for g in glb_r],
                                      [g["work"] for g in glb_r])  # fmt: skip
                a[f"work_gain_global_{m}"] = wg / r["work"]
                if r["wall"] and all(g["wall"] for g in glb_r):
                    tw, _ = interp_loglog(r[f"err_{m}"], [g[f"err_{m}"] for g in glb_r],
                                          [g["wall"] for g in glb_r])  # fmt: skip
                    a[f"wall_gain_global_{m}"] = tw / r["wall"]
            elif g_same is not None or glb_r:
                g_ref = g_same if g_same is not None else glb_r[0]
                a[f"work_gain_global_{m}"] = g_ref["work"] / r["work"]
                if r["wall"] and g_ref["wall"]:
                    a[f"wall_gain_global_{m}"] = g_ref["wall"] / r["wall"]
        a["work_gain"] = a["work_gain_semi"]
        a["wall_gain"] = a.get("wall_gain_semi")
        a["wall_gain_uniform"] = a.get("wall_gain_semi")
        a["work_gain_global"] = a.get("work_gain_global_semi")
        a["wall_gain_global"] = a.get("wall_gain_global_semi")
        a["wall_gain_vs_global"] = a.get("wall_gain_global_semi")
        r.update(a)
        out["arms"].append(a)
    out["uniform"] = [{k: u.get(k) for k in ("h_fine", "unknowns", "dt", "n_steps", "work",
                                             "err_semi", "err_td", "probe_agreement_ratio",
                                             "s_per_step", "wall", "wall_solverfit3d")}
                      for u in uni]  # fmt: skip
    if max_ratio > 0.2:
        raise RuntimeError(
            f"Probes disagree by {max_ratio:.4f}x the claimed error (> 0.2)"
        )
    return rows, out


STY = {"uniform": dict(c="k", m="o", label="uniform $h$ (SolverFIT3D discretisation)"),
       "graded": dict(c="tab:green", m="P", label="graded $h$ around cube (SolverFIT3D)"),
       ("global", 2): dict(c="tab:blue", m="s", label="2:1 patch, global $\\Delta t$ (wave 1)"),
       ("global", 3): dict(c="tab:cyan", m="D", label="3:1 patch, global $\\Delta t$ (wave 1)"),
       ("lts", 2): dict(c="tab:red", m="^", label="2:1 patch, LTS (this work)"),
       ("lts", 3): dict(c="tab:orange", m="v", label="3:1 patch, LTS (this work)")}  # fmt: skip


def _style(r):
    return (
        STY[r["kind"]]
        if r["kind"] in ("uniform", "graded")
        else STY[(r["kind"], r["ratio"])]
    )


def plot_cost(cases, path):
    if isinstance(cases, dict):
        rows, _ = analyse_cost(cases)
        cases = [("large patch: $\\varepsilon_r=9$ cube, 0.3 m", rows)]
    elif cases and isinstance(cases[0], dict):
        cases = [("large patch: $\\varepsilon_r=9$ cube, 0.3 m", cases)]
    else:
        cases = [
            (name, rows)
            for name, rows in cases
            if "small" not in name and all(r.get("case") != "small" for r in rows)
        ][:1]
    fig, axs = plt.subplots(
        len(cases), 2, figsize=(11, 4.3 * len(cases)), squeeze=False
    )
    for row, (name, rows) in enumerate(cases):
        for col, (xkey, xlabel) in enumerate((("work", "work = (edge + face updates) to $T$"),
                                              ("wall", "wall clock to $T$, 1 core [s]"))):  # fmt: skip
            ax = axs[row, col]
            groups = {}
            for r in rows:
                key = (
                    r["kind"]
                    if r["kind"] in ("uniform", "graded")
                    else (r["kind"], r["ratio"])
                )
                groups.setdefault(key, []).append(r)
            for key, g in groups.items():
                g = sorted(g, key=lambda r: r[xkey] or 0)
                if any(r[xkey] is None for r in g):
                    continue
                st = _style(g[0])
                ax.loglog([r[xkey] for r in g], [r["err_semi"] for r in g], marker=st["m"],
                          color=st["c"], label=st["label"], lw=1.5, ms=6)  # fmt: skip
            ax.set_xlabel(xlabel)
            ax.set_ylabel("relative error of $f_1$ (spatial, semi-discrete)")
            ax.grid(True, which="both", alpha=0.3)
            ax.set_title(f"({'ab'[col]}) {name}")
            if row == 0 and col == 0:
                ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_cpml(resF, path):
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.3))
    ax = axs[0]
    lab = {"nopatch": "no patch", "patch_global": "2:1 patch 1 coarse cell from layer, global $\\Delta t$",
           "patch_lts": "same, LTS", "through": "patch through layer, CFS $\\alpha/2\\pi$ = 1 GHz"}  # fmt: skip
    for v, mk in (
        ("nopatch", "o"),
        ("patch_global", "s"),
        ("patch_lts", "^"),
        ("through", "D"),
    ):
        g = sorted(
            [r for r in resF["cpml"] if r["variant"] == v], key=lambda r: r["n_pml"]
        )
        ax.semilogy(
            [r["n_pml"] for r in g], [r["err"] for r in g], marker=mk, label=lab[v]
        )
    ax.set_xlabel("layer thickness [coarse cells]")
    ax.set_ylabel("max |open - reference| / max |reference|")
    ax.set_title("(a) CPML reflection error at the probes")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7)
    ax = axs[1]
    r = [r for r in resF["cpml"] if r["variant"] == "patch_lts" and r["n_pml"] == 6]
    if r:
        r = r[0]
        t = np.array(r["t"]) * 1e9
        ax.plot(t, r["sig_ref"], "k-", lw=2, label="closed reference (3.7x larger)")
        ax.plot(t, r["sig_open"], "r--", lw=1.2, label="CPML, 6 cells, LTS")
        ax.plot(t, 1e3 * (np.array(r["sig_open"]) - np.array(r["sig_ref"])), "b-", lw=0.8,
                label="difference x 1000")  # fmt: skip
        ax.set_xlabel("time [ns]")
        ax.set_ylabel("$E_z$ at probe 2.5 cells from layer [V/m]")
        ax.set_title("(b) probe signal")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    ax = axs[2]
    for g in resF["growth"]:
        y = np.array(g["max_e"])
        t = (np.arange(len(y)) + 1) * g["every"] * g["dt"] * 1e9
        ax.semilogy(t, y, label=f"$\\alpha/2\\pi$ = {g['alpha_hz'] / 1e9:g} GHz")
    ax.set_xlabel("time [ns]")
    ax.set_ylabel("max $|e|$ / initial (random fields)")
    ax.set_title("(c) patch inside the layer: late-time growth")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def analyse_gpu(resG, rowsD):
    by = {(r["kind"], round(r["h_coarse"], 6), r["ratio"]): r for r in rowsD}
    out = []
    for g in resG["gpu"]:
        r = by[(g["kind"], round(g["h_coarse"], 6), g["ratio"])]
        row = dict(g)
        row["err_semi"] = r["err_semi"]
        row["err_td"] = r["err_td"]
        row["n_steps"] = r["n_steps"]
        if g["kind"] == "uniform":
            # SolverFIT3D runs at its own CFL; use its dt for the step count
            nst = r["n_steps"] * r["dt"] / g["solverfit3d_dt"]
            row["t_solverfit3d"] = g["solverfit3d"] * nst
            row["t_solverfit3d_fused"] = g["solverfit3d_fused"] * nst
        row["t_subgrid_block"] = g["subgrid_block"] * r["n_steps"]
        row["t_subgrid_csr"] = g["subgrid_csr"] * r["n_steps"]
        out.append(row)
    return out


def plot_gpu(rows, path):
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    uni = sorted(
        [r for r in rows if r["kind"] == "uniform"], key=lambda r: r["t_solverfit3d"]
    )
    ax.loglog([r["t_solverfit3d"] for r in uni], [r["err_semi"] for r in uni], "k-o",
              label="SolverFIT3D GPU (CSR), uniform $h$")  # fmt: skip
    ax.loglog([r["t_solverfit3d_fused"] for r in uni], [r["err_semi"] for r in uni], "k:o",
              mfc="w", label="SolverFIT3D GPU (fused kernels), uniform $h$")  # fmt: skip
    for kind, ratio in (("global", 2), ("lts", 2), ("global", 3), ("lts", 3)):
        g = sorted([r for r in rows if r["kind"] == kind and r["ratio"] == ratio],
                   key=lambda r: r["t_subgrid_block"])  # fmt: skip
        if not g:
            continue
        st = STY[(kind, ratio)]
        ax.loglog([r["t_subgrid_block"] for r in g], [r["err_semi"] for r in g], marker=st["m"],
                  color=st["c"], label=st["label"] + ", block kernels")  # fmt: skip
    ax.set_xlabel("GPU time to $T$ on one L4 [s]")
    ax.set_ylabel("relative error of $f_1$ (spatial, semi-discrete)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    os.makedirs(IMG, exist_ok=True)
    summary = {}
    cases = []
    rowsD = None
    for part, name in (("D", "large patch: $\\varepsilon_r=9$ cube, 0.3 m"),
                       ("E", "small patch: $\\varepsilon_r=9$ cube, 0.1 m")):  # fmt: skip
        res = load(part)
        if res is None:
            continue
        rows, out = analyse_cost(res)
        summary[f"cost_{part}"] = out
        cases.append((name, rows))
        if part == "D":
            rowsD = rows
    if cases:
        plot_cost(cases, os.path.join(IMG, "subgrid3d_lts_cost.png"))
    resF = load("F")
    if resF is not None:
        summary["cpml"] = [{k: r[k] for k in ("n_pml", "variant", "err", "err_probes", "dt")}
                           for r in resF["cpml"]]  # fmt: skip
        summary["growth"] = [{"alpha_hz": g["alpha_hz"], "final": g["max_e"][-1],
                              "max": max(g["max_e"]), "steps": len(g["max_e"]) * g["every"]}
                             for g in resF["growth"]]  # fmt: skip
        plot_cpml(resF, os.path.join(IMG, "subgrid3d_lts_cpml.png"))
    resG = load("G") or load("D")
    if resG and "gpu" in resG and "gpu_vs_cpu" in resG and rowsD is not None:
        g = analyse_gpu(resG, rowsD)
        summary["gpu"] = g
        summary["gpu_vs_cpu"] = resG["gpu_vs_cpu"]
        plot_gpu(g, os.path.join(IMG, "subgrid3d_lts_gpu.png"))
    with open(os.path.join(HERE, "summary_lts.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1)[:6000])


if __name__ == "__main__":
    main()
