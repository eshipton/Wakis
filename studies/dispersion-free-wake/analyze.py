"""
Figures and summary tables of the dispersion-free-wake study.

Reads results/*.json (run_case.py) and results/bench*.jsonl (bench_step.py),
writes figures/*.png and results/summary.json.
"""

import glob
import json
import os
import re
import sys

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analytic_box import box_modes, long_range_wake, loss_factor  # noqa: E402

LAB = {"tete0.99": "TE/TM (c dt = dz)", "yee0.99": "Yee, cfln 0.99",
       "yee0.5": "Yee, cfln 0.5 (Wakis default)"}  # fmt: skip
COL = {"tete0.99": "C3", "yee0.99": "C0", "yee0.5": "C2"}


def load():
    runs = {}
    for f in glob.glob("results/*.json"):
        name = os.path.basename(f)[:-5]
        if name in ("summary", "dispersion_vs_angle"):
            continue
        with open(f) as fh:
            runs[name] = json.load(fh)
    return runs


def key(r):
    a = r["args"]
    return f"{a['scheme']}{a['cfln']}"


def work(r):
    """Cell updates (cells x steps)."""
    return r["cells"] * r["Nt"]


def pipe_figure(runs, summary):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    out = {}
    for shape, mk in (("sq", "o"), ("rd", "s")):
        for sig, ls in ((1.6e-3, "-"), (0.8e-3, "--")):
            for sch in LAB:
                rs = [r for n, r in runs.items() if n.startswith(f"pipe_{shape}_")
                      and abs(r["args"]["sigma"] - sig) < 1e-9 and key(r) == sch]  # fmt: skip
                if not rs:
                    continue
                rs.sort(key=lambda r: r["args"]["length"])
                L = [r["args"]["length"] for r in rs]
                W = [np.max(np.abs(r["WP_interp"])) for r in rs]
                k = [abs(r["k_loss_interp"]) for r in rs]
                out[f"{shape}_s{sig}_{sch}"] = dict(L=L, maxW=W, k=k)
                lab = f"{LAB[sch]}, {'round' if shape == 'rd' else 'square'}, sigma={sig * 1e3:.1f} mm"
                ax[0].loglog(L, np.maximum(W, 1e-14), mk + ls, color=COL[sch],
                             label=lab, ms=4)  # fmt: skip
                ax[1].loglog(L, np.maximum(k, 1e-14), mk + ls, color=COL[sch], ms=4)
    for a_, t in zip(ax, ("max |W(s)| [V/pC]", "|loss factor| [V/pC]")):
        a_.set(xlabel="pipe length L [m]", ylabel=t)
        a_.set_xticks([0.05, 0.1, 0.2, 0.4, 0.8])
        a_.set_xticklabels(["0.05", "0.1", "0.2", "0.4", "0.8"])
        a_.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        a_.grid(True, which="both", alpha=0.3)
    ax[0].set_title("Smooth PEC pipe: spurious wake (exact: 0)\n" + r"$\Delta z$ = 0.4 mm, $\Delta x=\Delta y$ = 0.8 mm")
    ax[1].set_title("Spurious loss factor")
    ax[0].legend(fontsize=6.5, loc="center right")
    fig.tight_layout()
    fig.savefig("figures/pipe_spurious_wake.png", dpi=150)
    summary["pipe"] = out


def box_figure(runs, summary):
    a, d = 0.032, 0.096
    sigmas = sorted({r["args"]["sigma"] for n, r in runs.items() if n.startswith("box_")})
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    out = {}
    for sig, mk in zip(sigmas, "o^s"):
        om, kn = box_modes(a, a, d, a / 2, a / 2, sig)
        k_ref = loss_factor(om, kn, sig) * 1e-12
        # analytic long-range wake on a fine grid (ds = sigma/40), interpolated
        s_fine = np.arange(5 * sig, 0.31, sig / 40)
        W_fine = long_range_wake(s_fine, om, kn, sig) * 1e-12
        out[f"s{sig}"] = {"k_analytic": k_ref}
        for sch in LAB:
            rs = [r for n, r in runs.items() if n.startswith("box_")
                  and abs(r["args"]["sigma"] - sig) < 1e-9 and key(r) == sch]  # fmt: skip
            if not rs:
                continue
            rs.sort(key=lambda r: -r["args"]["dz"])
            res = [sig / r["args"]["dz"] for r in rs]
            # Wakis: negative loss factor = energy loss; analytic: positive
            err = [abs(-r["k_loss_interp"] / k_ref - 1) for r in rs]
            err_int = [abs(-r["k_loss"] / k_ref - 1) for r in rs]
            # long-range wake: RMS over s in [5 sigma, wakelength - 1 cm]
            werr = []
            for r in rs:
                s = np.asarray(r["s"])
                W = -np.asarray(r["WP_interp"])
                m = (s > 5 * sig) & (s < s.max() - 0.01)
                Wa = np.interp(s[m], s_fine, W_fine)
                werr.append(float(np.sqrt(np.mean((W[m] - Wa) ** 2))
                                  / np.sqrt(np.mean(Wa**2))))  # fmt: skip
            wk = [work(r) for r in rs]
            tl = [r["t_loop"] for r in rs]
            out[f"s{sig}"][sch] = dict(sigma_over_dz=res, k_err=err, k_err_wakis_int=err_int,
                                       wake_rms_err=werr, work=wk, t_loop=tl,
                                       k=[-r["k_loss_interp"] for r in rs],
                                       k_wakis_int=[-r["k_loss"] for r in rs])  # fmt: skip
            lab = f"{LAB[sch]}, sigma={sig * 1e3:.1f} mm"
            ax[0].loglog(res, err, mk + "-", color=COL[sch], label=lab, ms=4)
            ax[0].loglog(res, err_int, mk + ":", color=COL[sch], ms=3, alpha=0.4)
            ax[1].loglog(res, werr, mk + "-", color=COL[sch], ms=4)
            ax[2].loglog(wk, err, mk + "-", color=COL[sch], ms=4)
    x = np.array([2.5, 30])
    ax[0].loglog(x, 0.3 * (x / 2.5) ** -2, "k:", lw=0.8, label="order 2")
    ax[0].set(xlabel=r"$\sigma_z/\Delta z$ ($\Delta x=\Delta y=2\Delta z$)",
              ylabel="|k_loss / k_analytic - 1|",
              title="Closed box: loss factor vs analytic\n(solid: time-interpolated W; dotted: Wakis int(t/dt))")  # fmt: skip
    ax[1].set(xlabel=r"$\sigma_z/\Delta z$", ylabel="RMS wake error / RMS wake",
              title=r"Long-range wake, $s\in[5\sigma_z, s_{max}]$")  # fmt: skip
    ax[2].set(xlabel="work = cells x time steps", ylabel="|k_loss / k_analytic - 1|",
              title="Equal-work comparison")  # fmt: skip
    for a_ in ax:
        a_.grid(True, which="both", alpha=0.3)
    ax[0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig("figures/box_cavity_convergence.png", dpi=150)
    summary["box"] = out


def pillbox_figure(runs, summary):
    sigmas = sorted({r["args"]["sigma"] for n, r in runs.items() if n.startswith("pillbox_")})
    if not sigmas:
        return
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    out = {}
    for sig, mk in zip(sigmas, "o^sD"):
        rs_all = [r for n, r in runs.items() if n.startswith("pillbox_")
                  and abs(r["args"]["sigma"] - sig) < 1e-9]  # fmt: skip
        dzmin = min(r["args"]["dz"] for r in rs_all)
        refs = {key(r): -r["k_loss_interp"] for r in rs_all if r["args"]["dz"] == dzmin}
        if not refs:
            continue
        # Reference: finest TE/TM run (finest Yee shown for comparison)
        k_ref = refs.get("tete0.99", np.mean(list(refs.values())))
        out[f"s{sig}"] = {"k_ref": k_ref, "refs_finest": refs, "dz_ref": dzmin}
        for sch in ("tete0.99", "yee0.99"):
            rs = sorted([r for r in rs_all if key(r) == sch and r["args"]["dz"] > dzmin],
                        key=lambda r: -r["args"]["dz"])  # fmt: skip
            if not rs:
                continue
            res = [sig / r["args"]["dz"] for r in rs]
            err = [abs(-r["k_loss_interp"] / k_ref - 1) for r in rs]
            out[f"s{sig}"][sch] = dict(sigma_over_dz=res, k_err=err,
                                       k=[-r["k_loss_interp"] for r in rs],
                                       k_wakis_int=[-r["k_loss"] for r in rs],
                                       work=[work(r) for r in rs])  # fmt: skip
            ax[0].loglog(res, err, mk + "-", color=COL[sch], ms=4,
                         label=f"{LAB[sch]}, sigma={sig * 1e3:.1f} mm")  # fmt: skip
    ax[0].set(xlabel=r"$\sigma_z/\Delta z$", ylabel="|k / k_ref - 1|",
              title="Pillbox + pipes: loss factor vs finest TE/TM run")  # fmt: skip
    ax[0].grid(True, which="both", alpha=0.3)
    ax[0].legend(fontsize=6)
    # Wake potentials, sigma = 0.8 mm: coarse runs vs finest
    sig = 0.8e-3
    for n, r in sorted(runs.items()):
        if not n.startswith("pillbox_") or abs(r["args"]["sigma"] - sig) > 1e-9:
            continue
        dz = r["args"]["dz"]
        if dz not in (4e-4, 1e-4):
            continue
        s = np.asarray(r["s"]) * 1e3
        m = s < 60
        ax[1].plot(s[m], -np.asarray(r["WP_interp"])[m], color=COL[key(r)],
                   lw=1.4 if dz == 1e-4 else 0.8, ls="-" if dz == 1e-4 else "--",
                   label=f"{LAB[key(r)]}, dz={dz * 1e3:.1f} mm")  # fmt: skip
    ax[1].set(xlabel="s [mm]", ylabel="W(s) [V/pC] (positive = loss)",
              title=r"Pillbox wake potential, $\sigma_z$ = 0.8 mm")  # fmt: skip
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig("figures/pillbox_convergence.png", dpi=150)
    summary["pillbox"] = out


def bench_table(summary):
    rows = []
    for f in glob.glob("results/bench*.jsonl"):
        with open(f) as fh:
            rows += [json.loads(x) for x in fh if x.strip()]
    summary["bench"] = rows


def main():
    os.makedirs("figures", exist_ok=True)
    runs = load()
    summary = {"n_runs": len(runs)}
    pipe_figure(runs, summary)
    box_figure(runs, summary)
    pillbox_figure(runs, summary)
    bench_table(summary)
    with open("results/summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "bench"}, indent=1)[:6000])


if __name__ == "__main__":
    main()
