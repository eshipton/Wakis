"""
Subpixel-smoothed projection (SSP) study.

    python ssp_study.py offset       # Wakis FIT: objective and adjoint gradient vs
                                     # sub-cell interface offset (~20 s, 1 process)
    python ssp_study.py pixel        # pixel-level: profiles, slope ripple vs R,
                                     # convergence to binary (< 5 s)
    python ssp_study.py pilot DIR    # invrs-gym pilot (Cloud Batch results in DIR):
                                     # data/pilot.json, data/pilot.md, figures
    python ssp_study.py plot         # figures from data/*.json

Run from this folder with the Wakis fork on the path (PYTHONPATH=../..).
"""

import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).parent
DATA, FIGS = HERE / "data", HERE / "figures"


# --------------------------------------------------------------------------- A
def offset_sweep():
    """
    2D TM run (40 x 40 cells of 7.5 mm, CPML, point source at f0 = 3 GHz,
    eps 1 / 4): a straight level set rho_f = 1/2 + (x0 - x) / w (w = 0.1 m)
    in a 6 x 6 cm MaterialGrid (17 x 17 nodes) is moved from -1 to +1 cell.
    Objective F = 1e20 |Ez(f0)|^2 at a probe behind the design. dF/dx0 from
    the exact adjoint (sum of dF/drho / w). Arms at the E edges:
    SSP beta = inf, step projection (beta = inf, R -> 0), tanh beta = 64.
    """
    import wakis
    from wakis.inverse import AdjointSimulation, MaterialGrid
    from wakis.sources import CurrentSource, GaussianSource

    N, L, f0 = 40, 0.3, 3e9
    dx = L / N

    def solver():
        grid = wakis.GridFIT3D(-L / 2, L / 2, -L / 2, L / 2, -dx / 2, dx / 2, N, N, 1,
                               verbose=0)  # fmt: skip
        return wakis.SolverFIT3D(grid, bc_low=["cpml", "cpml", "periodic"],
                                 bc_high=["cpml", "cpml", "periodic"], bg=[1.0, 1.0, 0.0],
                                 n_pml=8, verbose=0)  # fmt: skip

    def sim(beta, radius):
        src = CurrentSource(GaussianSource(f0, 1e9), component="z", center=(-0.06, 0, 0))
        mon = {"probe": wakis.DFTMonitor([f0], components=["Ez"], center=(0.07, 0.01, 0))}
        design = MaterialGrid((0, 0, 0), (0.06, 0.06, 0.0), (17, 17, 1), 1.0, 4.0,
                              projection_beta=beta, projection_radius=radius)  # fmt: skip
        return AdjointSimulation(solver, [src], mon, design, 400)

    xs = np.linspace(-0.03, 0.03, 17)
    X = np.meshgrid(xs, xs, indexing="ij")[0][..., None]
    w = 0.1
    arms = {"ssp_inf": (np.inf, None), "step_inf": (np.inf, 1e-12), "tanh_64": (64.0, 1e-12)}
    offsets = np.linspace(-1, 1, 81) * dx + 1e-4 * dx  # avoid exact edge positions
    out = {"offsets_cells": (offsets / dx).tolist(), "dx_m": dx, "w_m": w}
    for name, (beta, radius) in arms.items():
        s = sim(beta, radius)
        F, dF = [], []
        for x0 in offsets:
            rho = 0.5 + (x0 - X) / w
            Ez = s.forward(rho)["probe"]["Ez"]
            F.append(1e20 * float(np.sum(np.abs(Ez) ** 2)))
            g = s.vjp({"probe": {"Ez": 2e20 * Ez}})
            dF.append(float(np.sum(g)) / w * dx)  # per cell of offset
        out[name] = {"F": F, "dF_dx0_per_cell": dF}
        print(name, "done", flush=True)
    # finite-difference check of the SSP slope at a few offsets
    s = sim(np.inf, None)
    checks = []
    for x0 in offsets[::20]:
        h = 1e-3 * dx
        Fp = 1e20 * np.sum(np.abs(s.forward(0.5 + (x0 + h - X) / w)["probe"]["Ez"]) ** 2)
        Fm = 1e20 * np.sum(np.abs(s.forward(0.5 + (x0 - h - X) / w)["probe"]["Ez"]) ** 2)
        checks.append({"x0_cells": x0 / dx, "fd_per_cell": float((Fp - Fm) / (2 * h) * dx)})
    out["ssp_fd_checks"] = checks
    DATA.mkdir(exist_ok=True)
    (DATA / "offset.json").write_text(json.dumps(out, indent=1))


# --------------------------------------------------------------------------- B
def pixel():
    from wakis.inverse import projection as ssp

    res = {}
    # (i) projection profiles (paper Fig. 4 analogue): rho_f = 1/2 + 1/2 tanh(x / 10R)
    R = 0.55
    x = np.linspace(-4, 4, 801) * R
    rho_f = 0.5 + 0.5 * np.tanh(x / (10 * R))
    g = np.abs(np.gradient(rho_f, x))
    res["profiles"] = {"x_over_R": (x / R).tolist(), "rho_f": rho_f.tolist()}
    for beta in (4.0, 16.0, 64.0, np.inf):
        out = ssp.smoothed_projection_pointwise(rho_f, g, beta, 0.5, R)[0]
        res["profiles"][f"ssp_{beta:g}"] = out.tolist()
        res["profiles"][f"tanh_{beta:g}"] = ssp.tanh_projection(rho_f, beta).tolist()
    # (ii) interface-motion slope dM/dx0 vs offset for several radii
    xs = np.arange(16.0)
    x0s = np.linspace(7, 9, 401) + 1e-6
    res["slope"] = {"x0_px": (x0s - 7).tolist()}
    for Rr in (0.45, 0.55, 0.75, 1.0):
        u = (xs[None, :] - x0s[:, None]) / Rr
        slope = np.sum(np.where(np.abs(u) < 1, 15 / 16 / Rr * (1 - u**2) ** 2, 0), axis=1)
        res["slope"][f"R{Rr:g}"] = slope.tolist()
        print(f"R = {Rr} px: slope min {slope.min():.3f} max {slope.max():.3f} "
              f"mean {slope.mean():.3f}")  # fmt: skip
    # (iii) convergence to binary (disk, filter width fixed in physical units)
    conv = {"N": [], "gray_inf": [], "l1_inf": [], "gray_tanh32": [], "l1_tanh32": []}
    for N in (32, 64, 128, 256, 512, 1024):
        h = 1 / N
        c = (np.arange(N) + 0.5) * h - 0.5
        Xg, Yg = np.meshgrid(c, c, indexing="ij")
        rf = 0.5 + (0.3 - np.hypot(Xg - 0.013, Yg + 0.021)) / 0.15
        b = rf > 0.5
        o = ssp.smoothed_projection(rf, np.inf, spacing=h)
        t = ssp.tanh_projection(rf, 32.0)
        conv["N"].append(N)
        conv["gray_inf"].append(float(np.mean((o > 0) & (o < 1))))
        conv["l1_inf"].append(float(np.abs(o - b).sum() * h * h))
        conv["gray_tanh32"].append(float(np.mean((t > 1e-3) & (t < 1 - 1e-3))))
        conv["l1_tanh32"].append(float(np.abs(t - b).sum() * h * h))
    res["convergence"] = conv
    DATA.mkdir(exist_ok=True)
    (DATA / "pixel.json").write_text(json.dumps(res))


# --------------------------------------------------------------------------- plots
def plot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGS.mkdir(exist_ok=True)
    colors = {"ssp_inf": "C0", "step_inf": "C3", "tanh_64": "C1"}
    labels = {"ssp_inf": "SSP, β = ∞", "step_inf": "step projection, β = ∞",
              "tanh_64": "tanh projection, β = 64"}  # fmt: skip
    if (DATA / "offset.json").exists():
        d = json.loads((DATA / "offset.json").read_text())
        x = np.array(d["offsets_cells"])
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for k in ("step_inf", "tanh_64", "ssp_inf"):
            ax[0].plot(x, d[k]["F"], color=colors[k], label=labels[k],
                       lw=2.2 if k == "ssp_inf" else 1.4)  # fmt: skip
            ax[1].plot(x, d[k]["dF_dx0_per_cell"], color=colors[k], label=labels[k],
                       lw=2.2 if k == "ssp_inf" else 1.4)  # fmt: skip
        fd = d["ssp_fd_checks"]
        ax[1].plot([c["x0_cells"] for c in fd], [c["fd_per_cell"] for c in fd], "ko",
                   mfc="none", label="SSP, finite difference")  # fmt: skip
        ax[0].set(xlabel="interface offset x0 / Δx", ylabel="objective F = |Ez(f0)|² (scaled)",
                  title="(a) objective vs sub-cell interface position")  # fmt: skip
        ax[1].set(xlabel="interface offset x0 / Δx", ylabel="dF/dx0 per cell (adjoint)",
                  title="(b) gradient seen by the optimizer")  # fmt: skip
        ax[1].axhline(0, color="k", lw=0.6)
        for a in ax:
            a.grid(alpha=0.3)
            a.legend(fontsize=8)
        fig.suptitle("Wakis FIT, 2D TM, 40×40 cells: straight level set moved through one design cell")
        fig.tight_layout()
        fig.savefig(FIGS / "a_offset_objective_gradient.png", dpi=150)
        plt.close(fig)
    if (DATA / "pixel.json").exists():
        d = json.loads((DATA / "pixel.json").read_text())
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        p = d["profiles"]
        xr = np.array(p["x_over_R"])
        ax[0].plot(xr, p["rho_f"], "k--", lw=1, label="filtered ρ̃")
        for i, b in enumerate(("4", "16", "64", "inf")):
            ax[0].plot(xr, p[f"ssp_{b}"], color=f"C{i}", label=f"SSP β = {b.replace('inf', '∞')}")
        ax[0].plot(xr, p["tanh_inf"], color="C3", ls=":", label="step (β = ∞)")
        ax[0].axvspan(-1, 1, color="0.9")
        ax[0].set(xlabel="x / R", ylabel="projected density",
                  title="(a) SSP profiles (R = 0.55 Δx, ρ̃ = ½ + ½ tanh(x/10R))")  # fmt: skip
        s = d["slope"]
        for i, Rr in enumerate(("0.45", "0.55", "0.75", "1")):
            ax[1].plot(s["x0_px"], s[f"R{Rr}"], color=f"C{i}", label=f"R = {Rr} Δx")
        ax[1].axhline(1, color="k", lw=0.8, ls="--")
        ax[1].set(xlabel="interface offset x0 / Δx", ylabel="dM/dx0 (material per Δx of motion)",
                  title="(b) β = ∞: slope ripple vs smoothing radius")  # fmt: skip
        c = d["convergence"]
        N = np.array(c["N"])
        ax[2].loglog(N, c["gray_inf"], "o-", color="C0", label="SSP β = ∞: gray-pixel fraction")
        ax[2].loglog(N, c["l1_inf"], "s-", color="C0", mfc="none", label="SSP β = ∞: L1 to binary")
        ax[2].loglog(N, c["gray_tanh32"], "o-", color="C1", label="tanh β = 32: gray fraction")
        ax[2].loglog(N, c["l1_tanh32"], "s-", color="C1", mfc="none", label="tanh β = 32: L1")
        ax[2].loglog(N, 2 * c["gray_inf"][0] * N[0] / N, "k:", label="O(Δx)")
        ax[2].set(xlabel="pixels across the unit square", ylabel="fraction / area",
                  title="(c) convergence to the binary design")  # fmt: skip
        for a in ax:
            a.grid(alpha=0.3, which="both")
            a.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(FIGS / "b_pixel_profiles_ripple_convergence.png", dpi=150)
        plt.close(fig)
    if (DATA / "pilot.json").exists():
        plot_pilot(json.loads((DATA / "pilot.json").read_text()), plt)


# --------------------------------------------------------------------------- pilot
def iterations_to_converge(log_records, frac=0.95):
    """First iteration at which the running best Wakis r=1 eval metric reaches
    `frac` of its total improvement over the run (from the first value)."""
    m = [r["eval_metric"] for r in log_records if "eval_metric" in r and "step" in r]
    if not m:
        return None
    best = np.maximum.accumulate(m)
    target = best[0] + frac * (best[-1] - best[0])
    return int(np.argmax(best >= target)) + 1


def collect_pilot(run_dir):
    run_dir = pathlib.Path(run_dir)
    rows = []
    for summ in sorted(run_dir.glob("shard_*/*/summary.json")):
        s = json.loads(summ.read_text())
        name = summ.parent.name
        arm = "ssp" if name.endswith("_ssp") else "round2"
        logs = [json.loads(line) for line in (summ.parent / "log.jsonl").read_text().splitlines()]
        sel = s.get("candidates", [{}])[s.get("selected_candidate", 0)] if s.get("candidates") else {}
        ls = s.get("length_scale_leaderboard_px")
        tgt = s.get("target_min_width_spacing_px")
        rows.append({
            "challenge": s["challenge"].replace("ceviche_lightweight_", ""),
            "seed": s["seed"], "arm": arm,
            "official": s.get("official_eval_metric"),
            "ceviche_10nm": s.get("ceviche_10nm_eval_metric"),
            "length_scale_px": ls, "target_px": tgt,
            "length_scale_pass": bool(ls and tgt and ls[0] >= tgt[0] and ls[1] >= tgt[1]),
            "iterations": sum(1 for r in logs if "step" in r and "loss" in r),
            "iters_to_95pct": iterations_to_converge(logs),
            "selected": s.get("selected_stage"), "selected_wakis": sel.get("score"),
            "opt_time_h": s.get("optimization_time_s", 0) / 3600,
            "errors": len(s.get("errors", [])),
            "trace": [r["eval_metric"] for r in logs if "step" in r and "loss" in r],
            "trace_stage": [r["stage"] for r in logs if "step" in r and "loss" in r],
            "dir": str(summ.parent.relative_to(run_dir)),
        })  # fmt: skip
    return rows


def design_evolution(run_dir, challenge="ceviche_lightweight_waveguide_bend", seed=0):
    """Figure: simulated density at the end of every stage (and a few steps
    inside the beta = inf stage) for both arms of one (challenge, seed)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_dir = pathlib.Path(run_dir)
    arms = {"round2": f"{challenge}_seed{seed}_robust", "ssp": f"{challenge}_seed{seed}_ssp"}
    picks = {}
    for arm, name in arms.items():
        hits = list(run_dir.glob(f"shard_*/{name}/snapshots.npz"))
        if not hits:
            continue
        z = np.load(hits[0])
        stage = z["stage"].astype(str)
        idx = [0]
        for k in range(1, len(stage)):
            if stage[k] != stage[k - 1]:
                idx.append(k - 1)
        idx.append(len(stage) - 1)
        picks[arm] = [(int(i), stage[i], int(z["step"][i]), float(z["eval_metric"][i]),
                       z["designs"][i].astype(float)) for i in idx]  # fmt: skip
    if not picks:
        return
    ncol = max(len(v) for v in picks.values())
    fig, ax = plt.subplots(len(picks), ncol, figsize=(2.3 * ncol, 3.0 * len(picks)), squeeze=False)
    for r, (arm, items) in enumerate(picks.items()):
        for c in range(ncol):
            a = ax[r, c]
            a.axis("off")
            if c >= len(items):
                continue
            i, st, step, m, d = items[c]
            lo, hi = d.min(), d.max()
            a.imshow(((d - lo) / max(hi - lo, 1e-12)).T, origin="lower", cmap="gray_r",
                     vmin=0, vmax=1)  # fmt: skip
            gray = np.mean((d > lo + 1e-3 * (hi - lo)) & (d < hi - 1e-3 * (hi - lo)))
            a.set_title(f"{arm}: it {i + 1}\n{st}:{step}\nWakis {m:+.3f}, gray {100 * gray:.0f} %",
                        fontsize=7)  # fmt: skip
    fig.suptitle(f"{challenge.replace('ceviche_lightweight_', '')}, seed {seed}: "
                 "simulated design at the end of each stage", fontsize=9)  # fmt: skip
    fig.tight_layout()
    FIGS.mkdir(exist_ok=True)
    fig.savefig(FIGS / f"c_design_evolution_{challenge.split('_')[-1]}_seed{seed}.png", dpi=150)
    plt.close(fig)


def pilot(run_dir):
    for ch in ("ceviche_lightweight_waveguide_bend", "ceviche_lightweight_mode_converter"):
        design_evolution(run_dir, ch, 0)
    rows = collect_pilot(run_dir)
    DATA.mkdir(exist_ok=True)
    (DATA / "pilot.json").write_text(json.dumps(rows, indent=1))
    lines = ["| challenge | arm | seed | official (40 nm) | ceviche 10 nm | length scale (target) "
             "| iterations | iters to 95 % | selected stage | opt. time h |", "|" + "---|" * 10]
    for r in sorted(rows, key=lambda r: (r["challenge"], r["arm"], r["seed"])):
        lines.append(
            f"| {r['challenge']} | {r['arm']} | {r['seed']} | {r['official']:+.3f} | "
            f"{(r['ceviche_10nm'] if r['ceviche_10nm'] is not None else float('nan')):+.3f} | "
            f"{r['length_scale_px']} ({r['target_px']}) | {r['iterations']} | "
            f"{r['iters_to_95pct']} | {r['selected']} | {r['opt_time_h']:.2f} |")  # fmt: skip
    lines.append("")
    for ch in sorted({r["challenge"] for r in rows}):
        for arm in ("round2", "ssp"):
            sub = [r for r in rows if r["challenge"] == ch and r["arm"] == arm]
            if not sub:
                continue
            off = np.array([r["official"] for r in sub], float)
            c10 = np.array([r["ceviche_10nm"] for r in sub if r["ceviche_10nm"] is not None], float)
            it = np.array([r["iters_to_95pct"] for r in sub], float)
            lines.append(
                f"- {ch} / {arm} (n = {len(sub)}): official mean {off.mean():+.3f} "
                f"(min {off.min():+.3f}, max {off.max():+.3f}); 10 nm mean "
                f"{c10.mean() if c10.size else float('nan'):+.3f}; length-scale pass "
                f"{sum(r['length_scale_pass'] for r in sub)}/{len(sub)}; "
                f"iterations to 95 % median {np.median(it):.0f}")  # fmt: skip
    (DATA / "pilot.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def plot_pilot(rows, plt):
    chs = sorted({r["challenge"] for r in rows})
    fig, ax = plt.subplots(1, 2 * len(chs), figsize=(5.5 * len(chs), 4))
    ax = np.atleast_1d(ax)
    for i, ch in enumerate(chs):
        for k, (key, title) in enumerate((("official", "official (40 nm)"),
                                          ("ceviche_10nm", "ceviche 10 nm"))):  # fmt: skip
            a = ax[2 * i + k]
            for j, arm in enumerate(("round2", "ssp")):
                sub = [r for r in rows if r["challenge"] == ch and r["arm"] == arm]
                vals = [r[key] for r in sub if r[key] is not None]
                a.scatter(np.full(len(vals), j) + np.linspace(-0.08, 0.08, len(vals)), vals,
                          color=f"C{j}", s=30)  # fmt: skip
                if vals:
                    a.hlines(np.mean(vals), j - 0.25, j + 0.25, color=f"C{j}")
            a.set_xticks([0, 1], ["round-2 (tanh + STE β=64)", "SSP (8, 32, ∞ + STE)"], fontsize=8)
            a.set(title=f"{ch}: {title}", ylabel="eval metric (higher is better)")
            a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "d_pilot_metrics.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(1, len(chs), figsize=(6 * len(chs), 4), squeeze=False)
    for i, ch in enumerate(chs):
        a = ax[0, i]
        for j, arm in enumerate(("round2", "ssp")):
            for r in (r for r in rows if r["challenge"] == ch and r["arm"] == arm):
                t = np.array(r["trace"], float)
                a.plot(np.arange(1, t.size + 1), t, color=f"C{j}", alpha=0.35, lw=0.8)
                a.plot(np.arange(1, t.size + 1), np.maximum.accumulate(t), color=f"C{j}", lw=1.6,
                       label=("round-2" if arm == "round2" else "SSP") if r["seed"] == 0 else None)
                st = r["trace_stage"]
                for k in range(1, len(st)):
                    if st[k] != st[k - 1] and r["seed"] == 0:
                        a.axvline(k + 0.5, color=f"C{j}", ls=":", lw=0.8)
        a.set(xlabel="iteration (simulated designs)", ylabel="Wakis r=1 eval metric",
              title=f"{ch}: per-iteration (thin) and running best (thick)")
        a.set_ylim(-1.0, 1.0)
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "e_pilot_convergence.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "plot"
    if cmd == "offset":
        offset_sweep()
    elif cmd == "pixel":
        pixel()
    elif cmd == "pilot":
        pilot(sys.argv[2])
    if cmd != "pilot" or len(sys.argv) > 2:
        plot()
