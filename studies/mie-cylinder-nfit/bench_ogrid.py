"""
O-grid curvilinear FIT arm on the shared benchmark suite (wakis.benchmarks):
B2 plasmonic nanocylinders (Au in n = 2 host, Ag in vacuum; TE and TM).
B3 Au dimers (gaps 2, 5, 10 nm, TE) with the multi-block
`BoxedCurvilinear2DArm` (O-grid per cylinder, boxes, Cartesian block).

Usage::

    python bench_ogrid.py list
    python bench_ogrid.py run --only "B2-Au-TE:ogrid:4e-9" --outdir DIR
    python bench_ogrid.py run --shard I --nshards N --outdir DIR

Every task is one simulation (one Cloud Batch task on its own machine).
"""

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
A_NANO = 20e-9
LADDER = [A_NANO / q for q in (5, 7, 10, 14, 20, 28, 40)]
CASES = ["B2-Au-TE", "B2-Au-TM", "B2-Ag-TE", "B2-Ag-TM"]


def arms():
    from wakis.benchmarks.arms.curvilinear2d import Curvilinear2DArm

    from wakis.benchmarks.arms.curvilinear2d import BoxedCurvilinear2DArm

    out = [Curvilinear2DArm(radial=1.0), Curvilinear2DArm(radial=0.5), BoxedCurvilinear2DArm()]
    return {a.name: a for a in out}


def tasks(phase="B2"):
    if phase == "B3":
        return [(c, "ogrid-boxed", h) for h in LADDER[::-1] for c in ("B3-g2", "B3-g5", "B3-g10")]
    return [(c, a, h) for h in LADDER[::-1] for c in CASES for a in ("ogrid", "ogrid-r0.5")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["list", "run"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--phase", default="B2")
    a = ap.parse_args()
    tl = tasks(a.phase)
    if a.mode == "list":
        for i, t in enumerate(tl):
            print(i, *t)
        return
    from wakis.benchmarks import get_case, run, save_record

    mine = ([(x.split(":")[0], x.split(":")[1], float(x.split(":")[2])) for x in a.only]
            if a.only else tl[a.shard :: a.nshards])  # fmt: skip
    a.outdir = a.outdir or str(HERE / "data" / a.phase)
    Path(a.outdir).mkdir(parents=True, exist_ok=True)
    A = arms()
    for case_name, arm_name, h in mine:
        rec = run(get_case(case_name), A[arm_name], h)
        save_record(rec, Path(a.outdir) / f"{case_name}_{arm_name}_{h * 1e9:.3f}nm.json")
        m = rec.get("metrics", {})
        print(case_name, arm_name, h, rec.get("error") or json.dumps(
            {k: m.get(k) for k in ("q_ext_nrms", "q_ext_lam_peak", "q_ext_height")}),
            rec.get("cost", {}).get("unknowns"), flush=True)  # fmt: skip


if __name__ == "__main__":
    main()
