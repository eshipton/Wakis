"""
Polygonal-FIT arms (hybrid squares + triangles, Voronoi polygons) on the
shared benchmark cases B2 (Au/Ag nanocylinder LSPR) and B3 (Au dimer gap).

Every task is one (case, arm, h) simulation, run as one Cloud Batch task on
its own n2-standard-2 machine; results are `wakis.benchmarks.run` records.
The baselines (Cartesian, triangles) are run by ../benchmarks/bench_study.py
with the same ladder.

    python poly_bench.py list
    python poly_bench.py run --shard I --nshards N --outdir DIR
    python poly_bench.py run --only "B2-Au-TE:poly-hybrid:4e-9" --outdir DIR
"""

import argparse
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
A_NANO = 20e-9
LADDER = [A_NANO / q for q in (5, 7, 10, 14, 20, 28, 40)]  # 4 nm .. 0.5 nm
CASES = {
    "poly-hybrid": ["B2-Au-TE", "B2-Au-TM", "B2-Ag-TE", "B2-Ag-TM", "B3-g2", "B3-g5", "B3-g10"],
    "poly-voronoi": ["B2-Au-TE", "B2-Au-TM", "B2-Ag-TE", "B2-Ag-TM"],
}


def arms():
    from wakis.benchmarks.arms.polygon2d import PolygonArm

    return {a.name: a for a in (PolygonArm("hybrid"), PolygonArm("voronoi"))}


def tasks():
    out = []
    for h in LADDER[::-1]:  # longest first
        for arm, cases in CASES.items():
            for c in cases:
                out.append((c, arm, h))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["list", "run"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--outdir", default=str(HERE / "data"))
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    tl = tasks()
    if a.mode == "list":
        for i, t in enumerate(tl):
            print(i, *t)
        return
    if a.only:
        sel = [(x.split(":")[0], x.split(":")[1], float(x.split(":")[2])) for x in a.only]
        mine = sel[a.shard :: a.nshards]
    else:
        mine = tl[a.shard :: a.nshards]
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from wakis.benchmarks import get_case, run, save_record

    A = arms()
    Path(a.outdir).mkdir(parents=True, exist_ok=True)
    for case, arm, h in mine:
        rec = run(get_case(case), A[arm], h)
        save_record(rec, Path(a.outdir) / f"{case}_{arm}_{h * 1e9:.3f}nm.json")
        m = rec.get("metrics", {})
        print(case, arm, h, rec.get("error") or {k: v for k, v in m.items() if not isinstance(v, (dict, list))},
              rec.get("cost"), flush=True)  # fmt: skip


if __name__ == "__main__":
    main()
