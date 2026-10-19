"""
Tables and convergence figure of the dispersive subpixel averaging study
(results of rod2d.py, rod_lorentz.py and sphere3d.py in runs/*.jsonl).

Mode identification: harmonic inversion finds every resonance in the band.
For the rods the reported mode is the strongest one (the symmetric source
and probe pattern makes the target mode dominant). For the sphere the TM_3
mode (+16.9 %) is excited as strongly as TM_1 by the z-dipole pattern, so
the strongest mode within +-2 % of the reference is reported (TM_2, the
nearest other exact mode, is at +4.7 %), or NaN if there is none.

Usage: python analyze.py runs/ out.png
"""

import json
import sys
from pathlib import Path

import numpy as np

TM3 = 5.294578119897345 / 4.5299507088545985 - 1
WINDOW = {"rod": np.inf, "lor": np.inf, "sph": 0.02}
LABEL = {
    "False": "staircase",
    "pec": "conformal PEC, staircased poles",
    "all": "dispersive Kottke diagonal (this work)",
    "tensor": "dispersive Kottke diagonal (this work)",
    "coupled": "edge-pixel full tensor (unstable, research)",
    "dev": "previous 'all' (eps_inf averaged, poles staircased)",
}


COLOR = {"False": "C0", "dev": "C1", "pec": "C2", "all": "C3", "coupled": "C4"}
# 'dev' and 'pec' give identical fields (eps_inf = 1 = vacuum): dashed on top
MARK = {"False": "o-", "dev": "s--", "pec": "o-", "all": "o-", "coupled": "^-"}


def select(modes, amps, window):
    modes, amps = np.asarray(modes), np.asarray(amps)
    near = np.abs(modes) < window
    if not near.any():
        return np.nan
    return modes[near][np.argmax(amps[near])]


def load(folder):
    rows = []
    for f in sorted(Path(folder).glob("*.jsonl")):
        kind = f.name.split("_")[0]
        dev = "dev" in f.name
        for line in f.open():
            r = json.loads(line)
            if kind == "rod":
                p = r["probes"][0]
                err, cells = select(p["modes"], p["amps"], WINDOW[kind]), r["cells"]
                case = "Drude rod"
            elif kind == "lor":
                err, cells = select(r["modes"], r["amps"], WINDOW[kind]), r["n"] ** 2
                case = "Lorentz rod"
            else:
                err, cells = select(r["modes"], r["amps"], WINDOW[kind]), r["cells"]
                case = f"Drude sphere ({r['case']})"
                if r["case"] == "cavity":
                    # TM_3 (k a = 5.294578, +16.88 %), dominant with 'all'
                    m3 = (1 + np.asarray(r["modes"])) / (1 + TM3) - 1
                    err3 = select(m3, r["amps"], 0.02)
                    conf3 = "dev" if dev else str(r["conformal"])
                    rows.append(("Drude sphere TM3", conf3, r["n"], cells, err3, None))
            conf = "dev" if dev else str(r["conformal"])
            rows.append((case, conf, r["n"], cells, err, r.get("err_im")))
    return rows


if __name__ == "__main__":
    rows = load(sys.argv[1])
    for row in sorted(rows):
        print("%-22s %-8s n=%4d cells=%9.0f err=%+.2e im=%s" % row)
    if len(sys.argv) > 2:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        cases = sorted({r[0] for r in rows if "open" not in r[0]})
        fig, axes = plt.subplots(1, len(cases), figsize=(5 * len(cases), 4))
        for ax, case in zip(np.atleast_1d(axes), cases):
            for conf in ("False", "dev", "pec", "all", "coupled"):
                sel = sorted((r[3], abs(r[4])) for r in rows
                             if r[0] == case and r[1] == conf and np.isfinite(r[4]))  # fmt: skip
                if sel:
                    x, y = np.array(sel).T
                    ax.loglog(x, y, MARK[conf], color=COLOR[conf], label=LABEL[conf])
            x = np.array([1e3, 1e5])
            dim = 3 if "sphere" in case else 2
            ax.loglog(x, 3e-3 * (x / 1e3) ** (-1 / dim), "k:", lw=1, label="order 1")
            ax.loglog(x, 3e-3 * (x / 1e3) ** (-2 / dim), "k--", lw=1, label="order 2")
            ax.set_xlabel("cells")
            ax.set_ylabel("|relative frequency error|")
            ax.set_title(case)
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(sys.argv[2], dpi=150)
