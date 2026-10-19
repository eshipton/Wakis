"""Publish subgrid B2/B3 records (data/<phase>/*.json) to the shared store.

python publish_sg.py --phase B2 --commit <sha> --job <batch job id>
"""

import argparse
import json
from pathlib import Path

from wakis.benchmarks.results import publish

HERE = Path(__file__).resolve().parent
RENAME = {"sg": "subgrid-2to1", "sg-lts4": "subgrid-2to1-lts4",
          "sg-gap4": "subgrid-2to1-gap4", "sg-gap4-lts8": "subgrid-2to1-gap4-lts8"}  # fmt: skip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True)
    ap.add_argument("--commit", required=True)
    ap.add_argument("--job", required=True)
    ap.add_argument("--tag-suffix", default="", help="e.g. '-stencil' (tag = <h>nm<suffix>)")
    a = ap.parse_args()
    for f in sorted((HERE / "data" / a.phase).glob("*.json")):
        rec = json.loads(f.read_text())
        if "error" in rec:
            print("skip (error)", f.name, rec["error"])
            continue
        arm = rec["arm"]
        rec["arm"] = RENAME.get(
            arm, "subgrid-2to1" + arm[2:] if arm.startswith("sg") else arm
        )
        rec["commit"] = a.commit
        rec["batch_job"] = a.job
        note = "wakis.subgrid 2:1 nested Cartesian, symmetric hanging-node interface"
        if "lts" in rec["arm"]:
            note += "; two-level LTS: steps are coarse steps, see cost.lts"
        tag = f"{rec['resolution'] * 1e9:.4f}nm{a.tag_suffix}" if a.tag_suffix else None
        print(publish(rec, tag=tag, machine="n2-standard-2 (spot, 1 task/VM)", notes=note))


if __name__ == "__main__":
    main()
