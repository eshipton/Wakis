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
        res.append(
            (float(np.average(f, weights=a)), float(np.median(f / (2 * g))), len(cl))
        )
    return res


def load():
    runs = {}
    for fn in sorted(glob.glob(os.path.join(DATA, "*.json"))):
        r = json.load(open(fn))
        if "found" not in r:
            continue
        r["gpu"] = bool(r.get("gpu", False) or fn.endswith("_gpu.json"))
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
                gpu=any(r.get("gpu", False) for r in recs),
            )
            table.setdefault(arm, []).append((h, ms, cost))
    # reference: BOR (r, z) eigensolver, Richardson of the 5 and 2.5 nm grids
    # (second order); TE0 family = the mode of each m with the smaller FSR
    bor = json.load(open(os.path.join(DATA, "bor_reference.json")))
    ref_f, lines = (
        [],
        [
            "| m | BOR 10 nm [THz] | 5 nm | 2.5 nm | Richardson [THz] | lambda [nm] |",
            "|---|---|---|---|---|---|",
        ],
    )
    # m = 15 is excluded from the metric: its TE0 mode anticrosses a
    # higher-order mode of the same m (BOR: 193.37 and 193.93 THz), and the
    # 3D runs resolve a single mode between the two
    ms_order = [k for k in sorted(bor, key=int) if k != "15"]
    for m in ms_order:
        f10, f5, f25 = (np.array(bor[m][k]) for k in ("10.0", "5.0", "2.5"))
        fr = (4 * f25 - f5) / 3
        # TE0: closest to the m-th member of the 7.8 THz-FSR family
        te0 = fr[np.argmin(np.abs(fr - (177.94e12 + 7.78e12 * (int(m) - 13))))]
        i = int(np.argmin(np.abs(fr - te0)))
        ref_f.append(te0)
        lines.append(
            f"| {m} | {f10[i] / 1e12:.4f} | {f5[i] / 1e12:.4f} | {f25[i] / 1e12:.4f} | "
            f"{te0 / 1e12:.4f} | {2.99792458e8 / te0 * 1e9:.2f} |"
        )
    ref = np.array(ref_f)
    cart = table.get("cart", [])
    prism = table.get("prism", [])

    def track(entries, fref):
        """Nearest high-Q mode to each reference (within 1.5 %), per run."""
        out = []
        for h, ms, cost in entries:
            fs = np.array([m[0] for m in ms])
            row = []
            for f in fref:
                j = np.argmin(np.abs(fs - f)) if fs.size else None
                row.append(
                    fs[j] if j is not None and abs(fs[j] / f - 1) < 0.015 else np.nan
                )
            out.append(row)
        return np.array(out)

    fc = track(cart, ref)
    fp = track(prism, ref)

    # (a) resonance frequency vs h
    fig_a, axs_a = plt.subplots(1, len(ms_order), figsize=(15, 4.4))
    for j, (ax, m, f_bor) in enumerate(zip(axs_a, ms_order, ref)):
        ax.axhline(
            f_bor / 1e12,
            color="k",
            ls="--",
            lw=1.0,
            label="BOR reference (r, z eigensolver)",
        )
        for arm, entries, F, col, mk in (
            ("cart", cart, fc, "C3", "^"),
            ("prism", prism, fp, "C0", "o"),
        ):
            if not entries:
                continue
            hs = np.array([e[0] for e in entries])
            g = np.array([e[2]["gpu"] for e in entries])
            lab = (
                "Wakis FIT (prisms, graded)"
                if arm == "prism"
                else "Wakis FIT (Cartesian conformal)"
            )
            ax.plot(hs, F[:, j] / 1e12, "-", color=col, marker=mk, label=lab)
            if g.any():
                ax.plot(
                    hs[g],
                    F[g, j] / 1e12,
                    "none",
                    color=col,
                    marker=mk,
                    mfc="w",
                    zorder=4,
                )
        ax.set_title(f"m = {m} ({2.99792458e8 / f_bor * 1e9:.2f} nm)")
        ax.set_xlabel("h [nm]")
        ax.set_ylabel("resonance frequency [THz]")
        ax.grid(True, which="both", alpha=0.3)
    axs_a[0].legend(fontsize=8)
    fig_a.suptitle(
        "Si micro-ring (R = 1.5 um, 500 x 220 nm, oxide): TE0 resonance frequency vs h"
    )
    fig_a.tight_layout()
    fig_a.savefig(os.path.join(FIG, "a_ring_spectrum_vs_h.png"), dpi=150)

    # (b) errors vs cost
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4))
    rows = [
        "| arm | h [nm] | unknowns | work | run [s] | max abs d-lambda [nm] | RMS rel. error |",
        "|---|---|---|---|---|---|---|",
    ]
    for arm, entries, F, col, mk in (
        ("cart", cart, fc, "C3", "^"),
        ("prism", prism, fp, "C0", "o"),
    ):
        if not entries:
            continue
        err = np.abs(F / ref - 1)
        dl = np.nanmax(err, 1) * LAM0 * 1e9
        rms = np.sqrt(np.nanmean(err**2, 1))
        N = np.array([e[2]["unknowns"] for e in entries])
        W = np.array([e[2]["work"] for e in entries])
        T = np.array([e[2]["run_s"] for e in entries])
        g = np.array([e[2]["gpu"] for e in entries])
        lab = (
            "Wakis FIT (prisms, graded)"
            if arm == "prism"
            else "Wakis FIT (Cartesian conformal)"
        )
        for k, (ax, x) in enumerate(zip(axs, (N, W, T))):
            idx = ~g if (k == 2 and g.any()) else slice(None)
            ax.loglog(x[idx], dl[idx], "-", color=col, marker=mk, label=lab)
            if g.any():
                ax.loglog(
                    x[g],
                    dl[g],
                    ":" if k == 2 else "none",
                    color=col,
                    marker=mk,
                    mfc="w",
                    zorder=4,
                    label="GPU (L4), time not comparable",
                )
        for (h, ms, cost), d, r in zip(entries, dl, rms):
            rows.append(
                f"| {arm} | {h:g} | {cost['unknowns']:,} | {cost['work']:.3g} | {cost['run_s']:.0f} | "
                f"{d:.3f} | {r:.2e} |"
            )
    for ax, xl in zip(
        axs,
        (
            "field unknowns (both symmetry runs)",
            "work = unknowns x steps",
            "run time [s]",
        ),
    ):
        ax.set_xlabel(xl)
        ax.set_ylabel(r"max $|\Delta\lambda|$ [nm]")
        ax.axhline(1.0, color="0.6", ls=":", lw=0.8)
        ax.axhline(0.1, color="0.6", ls=":", lw=0.8)
        ax.grid(True, which="both", alpha=0.3)
    axs[0].legend(loc="lower left", fontsize=8)
    fig.suptitle(
        "Si micro-ring (R = 1.5 um, 500 x 220 nm, oxide): resonance error vs cost"
    )
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "b_ring_error_vs_cost.png"), dpi=150)
    with open(os.path.join(DATA, "summary.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n\n" + "\n".join(rows) + "\n")
    print("\n".join(lines))
    print("\n".join(rows))


if __name__ == "__main__":
    main()
