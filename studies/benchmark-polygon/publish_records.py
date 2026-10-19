"""Publish this study's records to the shared benchmark store.

    python publish_records.py SRC_DIR BATCH_JOB COMMIT [--root ROOT]

Renames the arms to the store names (poly-hybrid -> hybrid-quadtri,
poly-voronoi -> voronoi-poly), copies the raw records into data/ and writes
the store layout under ROOT (default: the gs:// store, via
`wakis.benchmarks.results.publish`).
"""

import argparse
import glob
import json
import shutil
from pathlib import Path

from wakis.benchmarks.results import ROOT, publish

NAMES = {"poly-hybrid": "hybrid-quadtri", "poly-voronoi": "voronoi-poly"}
HERE = Path(__file__).resolve().parent

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("job")
ap.add_argument("commit")
ap.add_argument("--root", default=ROOT)
a = ap.parse_args()
n = 0
(HERE / "data").mkdir(exist_ok=True)
for f in sorted(glob.glob(f"{a.src}/**/*.json", recursive=True)):
    r = json.load(open(f))
    if r.get("arm") not in NAMES or "metrics" not in r:
        continue
    shutil.copy(f, HERE / "data" / Path(f).name)
    r["arm"] = NAMES[r["arm"]]
    r["commit"] = a.commit
    r["batch_job"] = a.job
    publish(r, root=a.root, machine="n2-standard-2 (spot, one run per machine)",
            notes="branch feature/polygon-fit; wakis/benchmarks/arms/polygon2d.py")  # fmt: skip
    n += 1
print(n, "records published to", a.root)
