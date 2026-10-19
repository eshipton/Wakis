"""
B4 (centre-fed thin-wire dipole, `wakis.benchmarks.dipole`) with the prism
FIT arm (`wakis.benchmarks.arms.prism_dipole.PrismDipoleArm`).

Usage::

    python dipole_study.py list
    python dipole_study.py run --task I --outdir out [--publish --machine n2-standard-2]
"""

import argparse
import json
import os
import sys

A_WIRE = 0.5e-3
LADDER = [A_WIRE / q for q in (1, 1.5, 2, 3, 4)]


def tasks():
    return [("B4", "prism-graded", h) for h in LADDER]


def main(argv=None):
    from wakis.benchmarks import get_case, run, save_record
    from wakis.benchmarks.arms.prism_dipole import PrismDipoleArm

    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["list", "run"])
    ap.add_argument("--task", type=int)
    ap.add_argument("--h", type=float)
    ap.add_argument("--outdir", default="out")
    ap.add_argument("--publish", action="store_true")
    ap.add_argument("--machine", default=None)
    args = ap.parse_args(argv)
    if args.cmd == "list":
        for i, t in enumerate(tasks()):
            print(i, *t)
        return
    case_name, arm_name, h = ("B4", "prism-graded", args.h) if args.h else tasks()[args.task]
    case = get_case(case_name)
    rec = run(case, PrismDipoleArm(name=arm_name), h, catch=False)
    if os.environ.get("WAKIS_COMMIT"):
        rec["commit"] = os.environ["WAKIS_COMMIT"]
    os.makedirs(args.outdir, exist_ok=True)
    fn = os.path.join(args.outdir, f"{case_name}_{arm_name}_{h * 1e9:.3f}nm.json")
    save_record(rec, fn)
    m = rec["metrics"]
    print(arm_name, h, {k: m.get(k) for k in ("f_res", "r_res", "z_nrms", "f_res_hz", "r_res_ohm")},
          {k: rec["cost"].get(k) for k in ("unknowns", "steps", "run_s")})  # fmt: skip
    if args.publish:
        from wakis.benchmarks.results import publish

        publish(rec, machine=args.machine, notes="prism FIT (PrismFIT3D), feature/prism-fit")


if __name__ == "__main__":
    sys.exit(main())
