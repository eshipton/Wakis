"""
Subgridded Cartesian FIT (`wakis.benchmarks.arms.subgrid2d`) on the shared
benchmark suite: B2 (Au / Ag nanocylinder LSPR, TE and TM) and B3 (Au
nanodimer, gaps 2/5/10 nm, gap-field enhancement).

Arms:

* ``sg``           -- 2:1 levels from H = h 2^L (<= lambda_host / 20, <= 16 h)
  down to h on and inside the metal, global time step;
* ``sg-lts4``      -- same mesh, two-level LTS with p = 4;
* ``sg-gap4``      -- B3: plus up to 2 extra levels in the gap (4 cells
  across the local gap width);
* ``sg-gap4-lts8`` -- same with LTS p = 8;
* phase ``B3z``: ``sg-zone2L1`` / ``sg-zone2L2`` -- the facing surfaces and
  the gap (local gap width < 2 x the case gap) get 1 or 2 more levels than
  the rest of the metal (global time step).

Tasks are (case, arm, h); one task per Cloud Batch machine (uncontended
timings). Records: `wakis.benchmarks.run` JSON in data/<phase>/.

Usage::

    python sg_bench.py list --phase B2
    python sg_bench.py run  --phase B2 --shard I --nshards N --outdir DIR
"""

import argparse
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
A_NANO = 20e-9
LADDER = [A_NANO / q for q in (5, 7, 10, 14, 20, 28, 40)]  # 4 .. 0.5 nm
PHASES = {
    "B2": ["B2-Au-TE", "B2-Au-TM", "B2-Ag-TE", "B2-Ag-TM"],
    "B3": ["B3-g5", "B3-g10", "B3-g2"],
    "B3z": ["B3-g5", "B3-g10", "B3-g2"],
    "B2p": ["B2-Au-TE", "B2-Au-TM", "B2-Ag-TE", "B2-Ag-TM"],
    "B3n": ["B3-g5", "B3-g10", "B3-g2"],  # 9-point gap metric (commit 6771d3c)
}


def arms(phase):
    from wakis.benchmarks.arms.subgrid2d import Subgrid2DArm

    if phase == "B2p":  # coarse grid at 40 cells per host wavelength
        a = Subgrid2DArm(ppw_out=40, name="sg-ppw40")
        return {a.name: a}
    if phase == "B3n":
        out = [Subgrid2DArm(), Subgrid2DArm(gap_zone=2, gap_levels=1),
               Subgrid2DArm(gap_zone=2, gap_levels=2)]  # fmt: skip
        return {a.name: a for a in out}
    if phase == "B3z":
        out = [
            Subgrid2DArm(gap_zone=2, gap_levels=1),
            Subgrid2DArm(gap_zone=2, gap_levels=2),
        ]
        return {a.name: a for a in out}
    out = [Subgrid2DArm(), Subgrid2DArm(lts=4)]
    if phase == "B3":
        out += [Subgrid2DArm(gap_cells=4), Subgrid2DArm(gap_cells=4, lts=8)]
    return {a.name: a for a in out}


def tasks(phase):
    out = []
    for case in PHASES[phase]:
        ladder = LADDER
        if case == "B3-g2" and phase not in ("B3z", "B3n"):
            ladder = [h for h in LADDER if h <= 1.0e-9 + 1e-15]
        for h in ladder[::-1]:
            for arm in arms(phase):
                if phase == "B3n" and "zone" in arm and h < 0.7e-9:
                    continue  # 0.125 nm gap cells: too many global steps
                out.append((case, arm, h))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["list", "run"])
    ap.add_argument("--phase", default="B2")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    tl = tasks(a.phase)
    if a.mode == "list":
        for i, t in enumerate(tl):
            print(i, *t)
        return
    from wakis.benchmarks import get_case, run, save_record

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    outdir = Path(a.outdir or HERE / "data" / a.phase)
    outdir.mkdir(parents=True, exist_ok=True)
    if a.only:
        mine = [
            (x.split(":")[0], x.split(":")[1], float(x.split(":")[2])) for x in a.only
        ]
    else:
        mine = tl[a.shard :: a.nshards]
    am = arms(a.phase)
    for case_name, arm_name, h in mine:
        case = get_case(case_name)
        rec = run(case, am[arm_name], h)
        name = outdir / f"{case_name}__{arm_name}__{h * 1e9:.3f}nm.json"
        save_record(rec, name)
        m = rec.get("metrics", {})
        print(case_name, arm_name, f"{h * 1e9:.2f} nm",
              rec.get("error") or {k: round(v, 4) for k, v in m.items()
                                   if isinstance(v, float)},
              json.dumps(rec.get("cost", {}).get("unknowns")), flush=True)  # fmt: skip


if __name__ == "__main__":
    main()
