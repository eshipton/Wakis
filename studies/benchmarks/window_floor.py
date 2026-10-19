"""
DFT-window floor of the B2/B3 metrics: the same arm and mesh at 1x and 3x the
case window (phase WIN of bench_study.py). The floor of a metric is the
difference between the two runs, measured with the metric itself (NRMS of
the 1x spectrum against the 3x one; relative peak/FWHM/enhancement changes).

    python window_floor.py [DATA_DIR]   -> data/window_floor.md
"""

import glob
import json
import sys
from pathlib import Path

import numpy as np

from wakis.benchmarks import get_case
from wakis.benchmarks import metrics as M

HERE = Path(__file__).resolve().parent


def main(d=HERE / "data" / "WIN"):
    recs = {}
    for f in glob.glob(str(Path(d) / "*.json")):
        r = json.loads(Path(f).read_text())
        if "metrics" not in r:
            print("error:", f, r.get("error"))
            continue
        recs[(r["case"], r["arm"], r.get("window_factor", 1))] = r
    lines = [
        "# DFT-window floor (1x vs 3x window, same arm and mesh)",
        "",
        "Floor = change of the metric when the window is tripled. Error vs reference "
        "at 1x and 3x for scale.",
        "",
        "| case | arm | window 1x [fs] | Q_ext NRMS (1x vs 3x) | peak lambda | peak height | "
        "FWHM | enh peak | Q_ext NRMS vs ref 1x / 3x | enh_peak vs ref 1x / 3x |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for (case_name, arm, wf), r1 in sorted(recs.items()):
        if wf != 1 or (case_name, arm, 3) not in recs:
            continue
        r3 = recs[(case_name, arm, 3)]
        case = get_case(case_name)
        lam = case.wavelengths
        s1, s3 = r1["metrics"]["spectra"], r3["metrics"]["spectra"]
        q1, q3 = np.array(s1["q_ext"]), np.array(s3["q_ext"])
        row = [case_name, arm, f"{r1['t_run'] * 1e15:.0f}", f"{M.nrms(q1, q3):.1e}"]
        p1, p3 = M.peak(lam, q1), M.peak(lam, q3)
        if p3["interior"]:
            row += [f"{abs(p1['lam_peak'] / p3['lam_peak'] - 1):.1e}",
                    f"{abs(p1['height'] / p3['height'] - 1):.1e}",
                    f"{abs(p1['fwhm'] / p3['fwhm'] - 1):.1e}" if np.isfinite(p3["fwhm"]) else "—"]  # fmt: skip
        else:
            row += ["—", "—", "—"]
        if "enh" in s1:
            e1, e3 = np.max(s1["enh"]), np.max(s3["enh"])
            row.append(f"{abs(e1 / e3 - 1):.1e}")
        else:
            row.append("—")
        m1, m3 = r1["metrics"], r3["metrics"]
        row.append(f"{m1['q_ext_nrms']:.1e} / {m3['q_ext_nrms']:.1e}")
        row.append(
            f"{m1['enh_peak']:.1e} / {m3['enh_peak']:.1e}" if "enh_peak" in m1 else "—"
        )
        lines.append("| " + " | ".join(row) + " |")
    out = HERE / "data" / "window_floor.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(*sys.argv[1:])
