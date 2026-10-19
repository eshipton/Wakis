"""Ring showcase: mode identification, Richardson reference, figures (data/*.json)."""

import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
FIG = os.path.join(HERE, "figures")
Q_MIN = 500.0
LAM0 = 1.55e-6


def modes(rec, rel=2e-4):
    """High-Q resonances of a run: clusters over probes -> (f, Q) per mode."""
    pts = [(f, g, a) for f, g, a in rec["found"] if g > 0 and f / (2 * g) > Q_MIN]
    pts.sort()
    out, cur = [], []
    for p in pts:
        if cur and p[0] - cur[-1][0] > rel * p[0]:
            out.append(cur)
            cur = []
        cur.append(p)
    if cur:
        out.append(cur)
    res = []
    for cl in out:
        f = np.array([p[0] for p in cl])
        g = np.array([p[1] for p in cl])
        a = np.array([p[2] for p in cl])
        res.append((float(np.average(f, weights=a)), float(np.median(f / (2 * g))), len(cl)))
    return res


def load():
    runs = {}
    for fn in sorted(glob.glob(os.path.join(DATA, "*.json"))):
        r = json.load(open(fn))
        key = r["arm"]
        runs.setdefault(key, {}).setdefault(r["h_nm"], []).append(r)
    return runs


def merged_modes(recs):
    """Cartesian quarter runs (PMC and PEC symmetry) together; prisms: one run."""
    ms = []
    for r in recs:
        ms += modes(r)
    ms.sort()
    return ms


def richardson(h, f, order=2.0):
    h1, h2 = h[-2], h[-1]
    f1, f2 = f[-2], f[-1]
    r = (h1 / h2) ** order
    return (r * f2 - f1) / (r - 1)


def main():
    os.makedirs(FIG, exist_ok=True)
    runs = load()
    table = {}
    for arm, by_h in runs.items():
        for h, recs in sorted(by_h.items(), reverse=True):
            ms = merged_modes(recs)
            cost = dict(
                unknowns=sum(r["unknowns"] for r in recs),
                work=sum(r["unknowns"] * r["steps"] for r in recs),
                run_s=sum(r["run_s"] for r in recs),
            )
            table.setdefault(arm, []).append((h, ms, cost))
    # reference modes: finest Cartesian run's high-Q modes; Richardson per mode
    cart = table.get("cart", [])
    ref_f = np.array([m[0] for m in cart[-1][1]])
    hs = np.array([t[0] for t in cart])

    def track(entries, fref):
        """Frequency of the mode nearest to each reference frequency, per run."""
        out = []
        for h, ms, cost in entries:
            fs = np.array([m[0] for m in ms])
            out.append([fs[np.argmin(np.abs(fs - f))] if fs.size else np.nan for f in fref])
        return np.array(out)

    fc = track(cart, ref_f)
    f_ref = richardson(hs, fc)
    prism = table.get("prism", [])
    fp = track(prism, ref_f)
    hp = np.array([t[0] for t in prism])
    f_ref_p = richardson(hp, fp) if len(prism) > 1 else np.full_like(f_ref, np.nan)
    lines = ["| mode | f_ref Cartesian (Richardson) [THz] | f_ref prism (Richardson) [THz] | difference [ppm] | lambda [nm] |",
             "|---|---|---|---|---|"]
    for i, f in enumerate(f_ref):
        lines.append(f"| {i} | {f / 1e12:.5f} | {f_ref_p[i] / 1e12:.5f} | {1e6 * (f_ref_p[i] / f - 1):.0f} | {2.99792458e8 / f * 1e9:.2f} |")
    ref = 0.5 * (f_ref + f_ref_p) if np.all(np.isfinite(f_ref_p)) else f_ref
    # errors
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4))
    rows = ["| arm | h [nm] | unknowns | work | run [s] | max abs d-lambda [nm] | RMS rel. error |", "|---|---|---|---|---|---|---|"]
    for arm, entries, F, col, mk in (("cart", cart, fc, "C3", "^"), ("prism", prism, fp, "C0", "o")):
        err = np.abs(F / ref - 1)
        dl = np.nanmax(err, 1) * LAM0 * 1e9
        rms = np.sqrt(np.nanmean(err**2, 1))
        N = np.array([e[2]["unknowns"] for e in entries])
        W = np.array([e[2]["work"] for e in entries])
        T = np.array([e[2]["run_s"] for e in entries])
        lab = "Wakis FIT (prisms, graded)" if arm == "prism" else "Wakis FIT (Cartesian conformal, uniform)"
        for ax, x in zip(axs, (N, W, T)):
            ax.loglog(x, dl, "-", color=col, marker=mk, label=lab)
        for (h, ms, cost), d, r in zip(entries, dl, rms):
            rows.append(f"| {arm} | {h:g} | {cost['unknowns']:,} | {cost['work']:.3g} | {cost['run_s']:.0f} | {d:.3f} | {r:.2e} |")
    for ax, xl in zip(axs, ("field unknowns", "work = unknowns x steps", "run time [s] (1 core)")):
        ax.set_xlabel(xl)
        ax.set_ylabel(r"max $|\Delta\lambda|$ of the ring resonances [nm]")
        ax.axhline(1.0, color="0.6", ls=":", lw=0.8)
        ax.axhline(0.1, color="0.6", ls=":", lw=0.8)
        ax.grid(True, which="both", alpha=0.3)
    axs[0].legend(fontsize=8)
    fig.suptitle("Si micro-ring (R = 1.5 um, 500 x 220 nm, oxide): resonance error vs cost")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "b_ring_error_vs_cost.png"), dpi=150)
    with open(os.path.join(DATA, "summary.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n\n" + "\n".join(rows) + "\n")
    print("\n".join(lines))
    print("\n".join(rows))


if __name__ == "__main__":
    main()
