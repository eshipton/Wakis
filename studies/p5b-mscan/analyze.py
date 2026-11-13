"""Post-process the P5 part B m-scan (run_point.py outputs in studies/p5b-mscan/out).

For each (qm, n_macro) group:
  * raw beta_q(m) and its Gauss-Newton sigma, relative to the modal prediction;
  * background-subtracted beta_q(m) = fit(I1(m) - I1(m=0)) when the matching m = 0 run exists
    (removes an m-independent coherent background, whose bias on the raw fit scales as 1/m);
  * DC depression delta_meas(m) - delta_disc, fitted as d0 + d2 m^2 (second-order <rho1 v1> term).
Writes summary.json and two figures (beta_q vs m, delta vs m^2) to studies/p5b-mscan/.
"""

import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "examples"))
import _space_charge_waves_common as C  # noqa: E402


def load(out):
    pts = []
    for js in sorted(glob.glob(os.path.join(out, "*.json"))):
        rec = json.load(open(js))
        rec["npz"] = js[:-5] + ".npz"
        pts.append(rec)
    return pts


def main(out=os.path.join(HERE, "out")):
    pts = [p for p in load(out) if not p.get("settle_factor")]
    extra = [p for p in load(out) if p.get("settle_factor")]
    groups = {}
    for p in pts:
        groups.setdefault((p["qm"], p["n_macro"]), []).append(p)
    summary = {}
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    for (qm, nm), g in sorted(groups.items()):
        g.sort(key=lambda p: p["m"])
        bg = next((p for p in g if p["m"] == 0), None)
        i1_bg = np.load(bg["npz"])["i1_part"] if bg else None
        params = C.compute_analytic_parameters(qm_scale=qm)
        rows = []
        for p in g:
            row = dict(m=p["m"], delta_meas=p["delta_meas"], delta_disc=p["delta_disc"])
            if p["m"] > 0:
                pred = p["bq_pred_part"]
                row.update(beta_raw=p["beta_q"], sig_raw=p["sigma_beta_q"],
                           rel_raw=p["beta_q"] / pred - 1.0, pred=pred, slack=p["slack_part"])
                if i1_bg is not None:
                    d = np.load(p["npz"])
                    fit = C.fit_beta_q_from_envelope(d["z"], d["i1_part"] - i1_bg, params["beta_q"],
                                                     z_min=p["z_min"], z_max=p["z_max"])
                    row.update(beta_sub=float(fit["beta_q"]), sig_sub=float(fit["sigma_beta_q"]),
                               rel_sub=float(fit["beta_q"]) / pred - 1.0)
            rows.append(row)
        ms = np.array([r["m"] for r in rows])
        dd = np.array([r["delta_meas"] - r["delta_disc"] for r in rows])
        A = np.vstack([np.ones_like(ms), ms**2]).T
        d0, d2 = np.linalg.lstsq(A, dd, rcond=None)[0]
        summary[f"qm{qm:g}_nm{nm}"] = dict(rows=rows, ddelta_fit_d0=float(d0), ddelta_fit_d2=float(d2))
        lab = f"q/m x{qm:g}, {nm}/step"
        mm = np.array([r["m"] for r in rows if r["m"] > 0])
        axs[0].errorbar(mm, [r["rel_raw"] for r in rows if r["m"] > 0],
                        yerr=[r["sig_raw"] / r["pred"] for r in rows if r["m"] > 0],
                        marker="o", capsize=3, label=lab + " raw")
        if i1_bg is not None:
            axs[0].errorbar(mm * 1.04, [r["rel_sub"] for r in rows if r["m"] > 0],
                            yerr=[r["sig_sub"] / r["pred"] for r in rows if r["m"] > 0],
                            marker="s", ls="--", capsize=3, label=lab + " minus m=0 run")
        axs[1].plot(ms**2, dd, "o-", label=lab)
    axs[0].axhline(0, color="k", lw=0.8)
    axs[0].set_xscale("log")
    axs[0].set_xlabel("modulation index m")
    axs[0].set_ylabel(r"$\beta_q^{\rm part}/\beta_q^{\rm modal} - 1$")
    axs[0].set_title(r"Particle $\beta_q$ vs modal prediction (coarse, sheet_lattice)")
    axs[0].legend(fontsize=7)
    axs[1].set_xlabel(r"$m^2$")
    axs[1].set_ylabel(r"$\delta_{\rm meas} - \delta_{\rm disc}$")
    axs[1].set_title("DC depression residual")
    axs[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "p5b_mscan.png"), dpi=150)
    summary["settle_variants"] = [
        dict(qm=p["qm"], m=p["m"], settle_factor=p["settle_factor"], beta=p["beta_q"],
             rel=p["beta_q"] / p["bq_pred_part"] - 1.0, sig=p["sigma_beta_q"]) for p in extra if p["m"] > 0]
    json.dump(summary, open(os.path.join(HERE, "summary.json"), "w"), indent=1)
    for k, v in summary.items():
        if k == "settle_variants":
            for s in v:
                print(f"settle qm={s['qm']:g} sf={s['settle_factor']:g}: rel={s['rel']:+.3e} sig={s['sig']:.2e}")
            continue
        print(f"== {k}: ddelta = {v['ddelta_fit_d0']:+.3e} + {v['ddelta_fit_d2']:+.3e} m^2")
        for r in v["rows"]:
            s = f"  m={r['m']:.5f} ddelta={r['delta_meas'] - r['delta_disc']:+.3e}"
            if "rel_raw" in r:
                s += f" raw={r['rel_raw']:+.3e}±{r['sig_raw'] / r['pred']:.1e} (slack {r['slack']:.1e})"
            if "rel_sub" in r:
                s += f" sub={r['rel_sub']:+.3e}±{r['sig_sub'] / r['pred']:.1e}"
            print(s)


if __name__ == "__main__":
    main(*sys.argv[1:])
