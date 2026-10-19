"""
Baseline arms of the shared benchmark suite (wakis.benchmarks).

Tasks are (case, arm, h) triples; every task is one simulation, run as one
Cloud Batch task on its own small machine (uncontended timings). Results are
JSON records (`wakis.benchmarks.run`) in data/<phase>/.

Usage::

    python bench_study.py list  --phase B2
    python bench_study.py run   --phase B2 --shard I --nshards N --outdir DIR
    python bench_study.py run   --phase B2 --only "B2-Au-TE:tri-graded:4e-9"
    python bench_study.py plot  --phase B2

Local runs only as smoke tests under bin/memguard.
"""

import argparse
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FIG = HERE / "figures"

A_NANO = 20e-9
LADDER = [A_NANO / q for q in (5, 7, 10, 14, 20, 28, 40)]  # 4 nm .. 0.5 nm


def arms():
    from wakis.benchmarks.arms.cartesian2d import Cartesian2DArm
    from wakis.benchmarks.arms.triangles2d import Triangles2DArm

    out = [
        Cartesian2DArm(conformal=None, graded=False),
        Cartesian2DArm(conformal=None, graded=True),
        Cartesian2DArm(conformal="all", graded=False),
        Cartesian2DArm(conformal="all", graded=True),
        Triangles2DArm(graded=False),
        Triangles2DArm(graded=True),
    ]
    return {a.name: a for a in out}


PHASES = {
    "B2": ["B2-Au-TE", "B2-Au-TM", "B2-Ag-TE", "B2-Ag-TM"],
    "B3": ["B3-g5", "B3-g10", "B3-g2"],
}


def tasks(phase):
    out = []
    names = list(arms())
    for case in PHASES[phase]:
        ladder = LADDER
        if case == "B3-g2":  # the gap needs h <= 1 nm
            ladder = [h for h in LADDER if h <= 1.0e-9 + 1e-15]
        for h in ladder[::-1]:  # longest first
            for arm in names:
                out.append((case, arm, h))
    return out


def run_tasks(mine, outdir, publish=False, machine=None):
    from wakis.benchmarks import get_case, run, save_record

    A = arms()
    Path(outdir).mkdir(parents=True, exist_ok=True)
    for case_name, arm_name, h in mine:
        case = get_case(case_name)
        rec = run(case, A[arm_name], h)
        tag = f"{case_name}_{arm_name}_{h * 1e9:.3f}nm"
        if os.environ.get("WAKIS_COMMIT"):
            rec["commit"] = os.environ["WAKIS_COMMIT"]
        save_record(rec, Path(outdir) / f"{tag}.json")
        if publish:
            from wakis.benchmarks.results import publish as pub

            pub(rec, machine=machine)
        m = rec.get("metrics", {})
        msg = rec.get("error") or " ".join(
            f"{k}={m[k]:.4g}"
            for k in ("q_ext_nrms", "q_ext_lam_peak", "enh_peak")
            if k in m
        )
        c = rec.get("cost", {})
        print(tag, msg, f"unknowns={c.get('unknowns')} steps={c.get('steps')} "
              f"run_s={c.get('run_s', 0):.0f}", flush=True)  # fmt: skip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["list", "run", "plot"])
    ap.add_argument("--phase", default="B2")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--only", nargs="*")
    ap.add_argument(
        "--publish",
        action="store_true",
        help="also upload every record to wakis.benchmarks.results.ROOT",
    )
    ap.add_argument("--machine", default=None)
    a = ap.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    tl = tasks(a.phase)
    if a.mode == "list":
        for i, t in enumerate(tl):
            print(i, *t)
        return
    if a.mode == "plot":
        sys.path.insert(0, str(HERE))
        from plot_bench import main_plot

        main_plot(a.phase)
        return
    if a.only:
        mine = [
            (s.split(":")[0], s.split(":")[1], float(s.split(":")[2])) for s in a.only
        ]
    else:
        mine = tl[a.shard :: a.nshards]
    run_tasks(mine, a.outdir or DATA / a.phase, a.publish, a.machine)


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"total {time.time() - t0:.0f} s", flush=True)
