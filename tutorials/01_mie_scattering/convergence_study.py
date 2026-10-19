"""
Resolution study for tutorial 01: Q_sca error versus cells per wavelength for
the staircase, conformal ('all') and tensor material models.

Each (resolution, model) pair is an independent run of
``mie_scattering.py --arm`` writing one JSON file to ``data/arms/``.

    # GPU arms run one at a time, so the measured wall times are comparable
    python convergence_study.py run --gpu --resolutions 10 15 20 30 40 50
    # CPU arms run in parallel, one process per core (memory permitting)
    python convergence_study.py run --cpu --jobs 12 --resolutions 10 15 20
    # merge the arms into data/convergence.json (read by the notebook)
    python convergence_study.py collect --machine "g2-standard-16, NVIDIA L4"

Every arm is checked with `wakis.estimate.preflight` before launch (inside
``mie_scattering.py``), and arms whose JSON already exists are skipped.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARMS = HERE / "data" / "arms"
MODELS = ("staircase", "all", "tensor")


def arm_path(res, model, backend):
    return ARMS / f"mie_{backend}_{res:g}_{model}.json"


def run_arm(res, model, backend):
    out = arm_path(res, model, backend)
    if out.exists():
        return f"skip {out.name}"
    log = out.with_suffix(".log")
    cmd = [
        sys.executable, str(HERE / "mie_scattering.py"), "--arm",
        "--resolution", f"{res:g}", "--conformal", model, f"--{backend}",
        "--out", str(out),
    ]  # fmt: skip
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    t0 = time.time()
    with open(log, "w") as fh:
        rc = subprocess.run(
            cmd, stdout=fh, stderr=subprocess.STDOUT, env=env
        ).returncode
    status = "ok" if rc == 0 and out.exists() else f"FAILED (rc={rc}, see {log.name})"
    return f"{out.name}: {status} in {time.time() - t0:.0f} s"


def run(args):
    ARMS.mkdir(parents=True, exist_ok=True)
    backend = "gpu" if args.gpu else "cpu"
    jobs = [(r, m) for r in args.resolutions for m in args.models]
    # Largest first, so the long runs start early
    jobs.sort(key=lambda j: -j[0])
    n_workers = 1 if backend == "gpu" else args.jobs
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        for msg in pool.map(lambda j: run_arm(j[0], j[1], backend), jobs):
            print(msg, flush=True)


def collect(args):
    runs = [json.loads(p.read_text()) for p in sorted(ARMS.glob("mie_*.json"))]
    if args.backend:
        runs = [
            r for r in runs if (r.get("gpu") is not None) == (args.backend == "gpu")
        ]
    runs.sort(key=lambda r: (r["conformal"], r["resolution_per_um"]))
    for r in runs:  # keep the committed file small: errors and timings only
        q_mie = r["q_mie"]
        r["dq"] = [round(a - b, 6) for a, b in zip(r["q_sca"], q_mie)]
        for key in ("q_sca", "q_mie", "q_abs", "x"):
            r[key] = [round(v, 6) for v in r[key]]
    out = HERE / "data" / "convergence.json"
    out.write_text(json.dumps({"machine": args.machine, "runs": runs}, indent=1))
    print(f"wrote {out} with {len(runs)} runs")
    for r in runs:
        print(
            f"{r['conformal']:>9} {r['resolution_per_um']:>4g}/µm "
            f"{r['cells_per_wavelength_in_sphere_at_xmax']:5.1f} cells/λ "
            f"err {r['rel_rms_error']:.4f}  {r['run_s'] + r['setup_s']:7.0f} s"
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    g = r.add_mutually_exclusive_group(required=True)
    g.add_argument("--gpu", action="store_true")
    g.add_argument("--cpu", action="store_true")
    r.add_argument("--resolutions", type=float, nargs="+", default=[10, 15, 20])
    r.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS)
    r.add_argument("--jobs", type=int, default=os.cpu_count())
    c = sub.add_parser("collect")
    c.add_argument("--machine", default="")
    c.add_argument("--backend", choices=["cpu", "gpu"], default=None)
    a = ap.parse_args()
    run(a) if a.cmd == "run" else collect(a)


if __name__ == "__main__":
    main()
