"""
LTS on the shared benchmark suite (wakis.benchmarks): B2 (Au/Ag
nanocylinder, Lorentz-Drude) and B3 (Au dimer, gaps 2/5/10 nm).

Arms: ``tri-graded`` (global leapfrog at the global minimum dt) and
``tri-graded-lts`` (same mesh, ``lts=2``: Lorentz-Drude cells inside F,
poles frozen over the coarse step). Same ladders as
``studies/benchmarks/bench_study.py``; one task per Cloud Batch machine.

Usage::

    python bench_lts.py list
    python bench_lts.py run --shard I --nshards N --outdir DIR
    python bench_lts.py run --only "B3-g5:tri-graded-lts:4e-9" --outdir DIR
"""

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "benchmarks"))
import bench_study as B  # noqa: E402


def arms():
    from wakis.benchmarks.arms.triangles2d import Triangles2DArm

    out = [Triangles2DArm(graded=True), Triangles2DArm(graded=True, lts=2)]
    return {a.name: a for a in out}


def tasks():
    out = []
    for case in B.PHASES["B3"] + B.PHASES["B2"]:
        ladder = B.LADDER
        if case == "B3-g2":
            ladder = [h for h in ladder if h <= 1.0e-9 + 1e-15]
        for h in ladder[::-1]:
            for arm in arms():
                out.append((case, arm, h))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["list", "run"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--outdir", default=str(HERE / "data" / "bench"))
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    tl = tasks()
    if a.mode == "list":
        for i, t in enumerate(tl):
            print(i, *t)
        return
    if a.only:
        mine = [(s.split(":")[0], s.split(":")[1], float(s.split(":")[2]))
                for s in a.only]  # fmt: skip
    else:
        mine = tl[a.shard :: a.nshards]
    B.arms = arms  # run_tasks looks the arms up here
    B.run_tasks(mine, a.outdir)


if __name__ == "__main__":
    main()
