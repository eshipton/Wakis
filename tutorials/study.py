"""
Parameter studies for the tutorials: run independent "arms" of a tutorial
script in parallel and merge their JSON summaries.

Every tutorial script accepts ``--arm --out FILE`` plus its own options
(e.g. ``--resolution``, ``--conformal``, ``--pol``) and then performs one
simulation (after its own `wakis.estimate` preflight) and writes a JSON
summary instead of the figures. This tool expands a grid of options into
arms and runs them:

    # CPU: one process per core (independent 2D runs are single-threaded)
    python tutorials/study.py run tutorials/01_mie_cylinder/mie_cylinder.py \\
        --backend cpu --jobs 14 \\
        --grid resolution=10,15,20,30,40 conformal=staircase,all,tensor pol=TM,TE
    # GPU: one arm at a time by default (comparable wall times); small,
    # latency-bound arms can share the GPU with --jobs N
    python tutorials/study.py run tutorials/01_mie_cylinder/mie_cylinder.py \\
        --backend gpu --grid resolution=60,80 conformal=staircase,all,tensor pol=TM,TE
    # merge data/arms/*.json into data/convergence.json (read by the notebook)
    python tutorials/study.py collect tutorials/01_mie_cylinder/mie_cylinder.py \\
        --machine "g2-standard-16 (8-core Xeon, NVIDIA L4)"

Arms whose JSON exists are skipped, so an interrupted study resumes.
"""

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def _arms_dir(script):
    return Path(script).resolve().parent / "data" / "arms"


def _expand(grid):
    keys, values = [], []
    for item in grid:
        key, vals = item.split("=", 1)
        keys.append(key)
        values.append(vals.split(","))
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def _name(params, backend):
    return "_".join([backend] + [f"{k}-{v}" for k, v in params.items()])


def run_arm(script, params, backend):
    out = _arms_dir(script) / f"{_name(params, backend)}.json"
    if out.exists():
        return f"skip {out.name}"
    cmd = [sys.executable, str(script), "--arm", f"--{backend}", "--out", str(out)]
    for k, v in params.items():
        cmd += [f"--{k}", v]
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    env["OPENBLAS_NUM_THREADS"] = "1"
    t0 = time.time()
    with open(out.with_suffix(".log"), "w") as fh:
        rc = subprocess.run(
            cmd, stdout=fh, stderr=subprocess.STDOUT, env=env
        ).returncode
    ok = rc == 0 and out.exists()
    status = "ok" if ok else f"FAILED rc={rc} (see {out.with_suffix('.log').name})"
    return f"{out.name}: {status} in {time.time() - t0:.0f} s"


def run(args):
    _arms_dir(args.script).mkdir(parents=True, exist_ok=True)
    arms = _expand(args.grid)
    key = args.sort_key
    if key:  # most expensive first, so the long runs start early
        arms.sort(key=lambda p: -float(p.get(key, 0)))
    workers = (
        args.jobs if args.jobs else (1 if args.backend == "gpu" else os.cpu_count())
    )
    failed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for msg in pool.map(lambda p: run_arm(args.script, p, args.backend), arms):
            print(msg, flush=True)
            failed += "FAILED" in msg
    if failed:
        sys.exit(f"{failed} arm(s) failed")


def collect(args):
    folder = _arms_dir(args.script)
    runs = [json.loads(p.read_text()) for p in sorted(folder.glob("*.json"))]
    out = folder.parent / "convergence.json"
    out.write_text(json.dumps({"machine": args.machine, "runs": runs}, indent=1))
    print(f"wrote {out} with {len(runs)} runs")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("script")
    r.add_argument("--backend", choices=["cpu", "gpu"], required=True)
    r.add_argument("--grid", nargs="+", required=True, help="key=v1,v2 ...")
    r.add_argument("--jobs", type=int, default=None,
                   help="parallel arms (default: all cores; 1 on the GPU)")  # fmt: skip
    r.add_argument("--sort-key", default="resolution")
    c = sub.add_parser("collect")
    c.add_argument("script")
    c.add_argument("--machine", default="")
    a = ap.parse_args()
    run(a) if a.cmd == "run" else collect(a)


if __name__ == "__main__":
    main()
