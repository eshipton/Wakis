"""
Quadtree + conforming triangle-band (``subtri-hybrid``) benchmark study on
cases B1 (dielectric cylinder), B2 (Au/Ag nanocylinder LSPR), and B3 (Au dimer
gap enhancement).

Every task is one ``(case, arm, h)`` simulation, run as one Cloud Batch task on
its own ``n2-standard-2`` machine; results are ``wakis.benchmarks.run`` records
written to ``<outdir>/<case>_<arm>_<h_nm:.3f>nm.json``.

Usage::

    python subtri_bench.py list --phase B2
    python subtri_bench.py run  --phase B2 --shard I --nshards N --outdir DIR
    python subtri_bench.py run  --phase B2 --smoke --only "B2-Au-TE:subtri-hybrid:4e-9"
"""

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DATA = HERE / "data"

A_NANO = 20e-9
LADDER = [A_NANO / q for q in (5, 7, 10, 14, 20, 28, 40)]  # 4 nm .. 0.5 nm

PHASES = {
    "B1": ("B1-TE", "B1-TM"),
    "B2": ("B2-Au-TE", "B2-Au-TM", "B2-Ag-TE", "B2-Ag-TM"),
    "B3": ("B3-g10", "B3-g5", "B3-g2"),
    # reruns of the B3 tasks whose gap points fell inside squares (RT diagnostic
    # raised before the RT fallback); B1 without LTS (attribution)
    "B3FIX": ("B3-g10", "B3-g5"),
    "B1NOLTS": ("B1-TE", "B1-TM"),
}


def arms():
    """Return the study arms keyed by arm name."""
    from wakis.benchmarks.arms.subtri2d import SubTriArm

    out = (
        SubTriArm(ratio=4, ppw_out=20, lts="auto", name="subtri-hybrid"),
        SubTriArm(ratio=4, ppw_out=20, lts=None, name="subtri-hybrid-nolts"),
        SubTriArm(ratio=2, ppw_out=None, lts="auto", name="subtri-hybrid-r2"),
        SubTriArm(ratio=4, ppw_out=None, lts="auto", name="subtri-hybrid-r4"),
        SubTriArm(ratio=2, ppw_out=None, lts=None, name="subtri-hybrid-r2-nolts"),
    )
    return {a.name: a for a in out}


def _b1_ladder(case_name):
    from wakis.benchmarks import get_case
    from wakis.benchmarks.arms.triangles2d import Triangles2DArm

    return Triangles2DArm(graded=True).resolutions(get_case(case_name))


def _expected_cost_key(task):
    """Sort key ordering tasks by expected computational cost descending."""
    case_name, arm_name, h = task
    gap_h = {"B3-g2": 0.5e-9, "B3-g5": 1.25e-9, "B3-g10": 2.5e-9}.get(case_name, h)
    eff_h = min(h, gap_h)
    # Within the same h, no-LTS (or smaller ratio r2) is more expensive than LTS (or r4).
    arm_rank = 0 if arm_name.endswith(("-nolts", "-r2")) else 1
    return (h, eff_h, arm_rank, case_name, arm_name)


def tasks(phase, smoke=False):
    """Return ``(case, arm, h)`` tasks for ``phase`` sorted finest-first."""
    phase = phase.upper()
    if phase not in PHASES:
        raise ValueError(f"Unknown phase {phase!r}; choose from {tuple(PHASES)}")

    out = []
    if phase in ("B1", "B1NOLTS"):
        arm_names = (("subtri-hybrid-r2", "subtri-hybrid-r4") if phase == "B1"
                     else ("subtri-hybrid-r2-nolts",))  # fmt: skip
        for case_name in PHASES[phase]:
            lad = _b1_ladder(case_name)
            if smoke:
                lad = [max(lad)]
            for h in lad:
                for arm_name in arm_names:
                    out.append((case_name, arm_name, float(h)))
    else:
        arm_names = ("subtri-hybrid", "subtri-hybrid-nolts")
        lad = [max(LADDER)] if smoke else list(LADDER)
        for case_name in PHASES[phase]:
            for h in lad:
                for arm_name in arm_names:
                    out.append((case_name, arm_name, float(h)))

    if phase == "B3FIX":
        hmax = {"B3-g10": 1.43e-9, "B3-g5": 0.72e-9}
        out = [t for t in out if t[2] <= hmax[t[0]]]
    out.sort(key=_expected_cost_key)
    return out


def run_tasks(mine, outdir, smoke=False):
    """Execute ``mine`` tasks and write JSON records to ``outdir``."""
    from wakis.benchmarks import get_case, run, save_record

    A = arms()
    Path(outdir).mkdir(parents=True, exist_ok=True)
    for case_name, arm_name, h in mine:
        case = get_case(case_name)
        if smoke:
            case.t_run = case.t_run / 8.0
        rec = run(case, A[arm_name], h)
        if smoke:
            rec["smoke"] = True
        tag = f"{case_name}_{arm_name}_{h * 1e9:.3f}nm"
        save_record(rec, Path(outdir) / f"{tag}.json")
        m = rec.get("metrics", {})
        msg = rec.get("error") or " ".join(
            f"{k}={m[k]:.4g}"
            for k in ("q_ext_nrms", "q_ext_lam_peak", "enh_peak", "enh_nrms")
            if k in m
        )
        c = rec.get("cost", {})
        lts = c.get("lts")
        p_lts = lts.get("p") if isinstance(lts, dict) else 1
        print(
            tag,
            msg,
            f"unknowns={c.get('unknowns')} steps={c.get('steps')} "
            f"p={p_lts} work={c.get('work')} run_s={c.get('run_s', 0):.2f}",
            flush=True,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["list", "run"])
    ap.add_argument("--phase", default="B2")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--only", nargs="*")
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="coarsest rung with t_run / 8 for fast local verification",
    )
    a = ap.parse_args()
    phase = a.phase.upper()
    tl = tasks(phase, smoke=a.smoke)
    if a.mode == "list":
        for i, t in enumerate(tl):
            print(i, *t)
        print(f"{len(tl)} tasks")
        return

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    if a.only:
        sel = [
            (s.split(":")[0], s.split(":")[1], float(s.split(":")[2])) for s in a.only
        ]
        mine = sel[a.shard :: a.nshards]
    else:
        mine = tl[a.shard :: a.nshards]
    outdir = a.outdir or str(DATA / phase)
    run_tasks(mine, outdir, smoke=a.smoke)


if __name__ == "__main__":
    main()
