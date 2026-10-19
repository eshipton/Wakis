"""
High-order DGTD arms on the shared benchmark suite (wakis.benchmarks):
B2 (Au / Ag nanocylinder LSPR) and B3 (Au dimer gap).

Arms (`wakis.highorder.bench_arm.DGTD2DArm`, box domain with CFS-PML):
dg2/dg3/dg4 (curved elements) and dg4-straight (polygonal interfaces: the
geometric interface error alone). Ladder: h = a / q on the cylinders,
q = 2, 3, 4, 6, 8 (a = 20 nm).

Usage::

    python ho_bench.py list --phase B2
    python ho_bench.py run --phase B2 --shard I --nshards N --outdir DIR
    python ho_bench.py plot --phase B2
"""

import argparse
import os
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
A_NANO = 20e-9
LADDER = [A_NANO / q for q in (2, 3, 4, 6, 8)]
PHASES = {
    "B2": ["B2-Au-TE", "B2-Au-TM", "B2-Ag-TE", "B2-Ag-TM"],
    "B3": ["B3-g5", "B3-g10", "B3-g2"],
}


def arms():
    from wakis.highorder.bench_arm import DGTD2DArm

    out = [DGTD2DArm(p=2), DGTD2DArm(p=3), DGTD2DArm(p=4),
           DGTD2DArm(p=4, curved=False)]  # fmt: skip
    return {a.name: a for a in out}


def tasks(phase):
    out = []
    for case in PHASES[phase]:
        for h in LADDER[::-1]:
            for arm in arms():
                out.append((case, arm, h))
    return out


def run_tasks(mine, outdir):
    from wakis.benchmarks import get_case, run, save_record

    A = arms()
    Path(outdir).mkdir(parents=True, exist_ok=True)
    for case_name, arm_name, h in mine:
        rec = run(get_case(case_name), A[arm_name], h)
        tag = f"{case_name}_{arm_name}_{h * 1e9:.3f}nm"
        save_record(rec, Path(outdir) / f"{tag}.json")
        m = rec.get("metrics", {})
        msg = rec.get("error") or " ".join(
            f"{k}={m[k]:.4g}" for k in ("q_ext_nrms", "q_ext_lam_peak", "enh_peak") if k in m
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
    a = ap.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    tl = tasks(a.phase)
    if a.mode == "list":
        for i, t in enumerate(tl):
            print(i, *t)
        return
    if a.mode == "plot":
        from plot_bench_ho import main_plot

        main_plot(a.phase)
        return
    if a.only:
        mine = [(s.split(":")[0], s.split(":")[1], float(s.split(":")[2])) for s in a.only]
    else:
        mine = tl[a.shard :: a.nshards]
    run_tasks(mine, a.outdir or DATA / a.phase)


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"total {time.time() - t0:.0f} s", flush=True)
