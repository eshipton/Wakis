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
        Triangles2DArm(graded=False, n_pml=40),
        Triangles2DArm(graded=True, n_pml=40),
    ]
    from wakis.benchmarks.arms.cartesian_dipole import CartesianDipoleArm

    gpu = os.environ.get("BENCH_GPU") == "1"
    out += [
        CartesianDipoleArm(conformal=False),  # legacy STL cell masks
        CartesianDipoleArm(conformal="pec"),
        CartesianDipoleArm(conformal="edges", use_gpu=gpu),
        CartesianDipoleArm(conformal="pec", pad_xy=0.42, pad_z=0.28,
                           name="cart-conformal-pec-graded-bigdomain"),
        CartesianDipoleArm(conformal="edges", pad_xy=0.42, pad_z=0.28, use_gpu=gpu,
                           name="cart-staircase-edges-graded-bigdomain"),
        CartesianDipoleArm(conformal="edges", ppw_out=40, use_gpu=gpu,
                           name="cart-staircase-edges-graded-ppw40"),
        CartesianDipoleArm(conformal="edges", hz_ratio=1.5, use_gpu=gpu,
                           name="cart-staircase-edges-graded-hz1.5ht"),
        CartesianDipoleArm(conformal="edges", hz_ratio=0.75, use_gpu=gpu,
                           name="cart-staircase-edges-graded-hz0.75ht"),
        CartesianDipoleArm(conformal="edges", port="sheet", use_gpu=gpu),
        CartesianDipoleArm(conformal="edges", ends="round", use_gpu=gpu),
        CartesianDipoleArm(conformal="edges", port="sheet", ends="round", use_gpu=gpu),
    ]  # fmt: skip
    return {a.name: a for a in out}


PHASES = {
    "B2": ["B2-Au-TE", "B2-Au-TM", "B2-Ag-TE", "B2-Ag-TM"],
    "B3": ["B3-g5", "B3-g10", "B3-g2"],
    "B3m2": ["B3-g5", "B3-g10", "B3-g2"],
    "B4": ["B4"],
    "B4b": ["B4"],
    "B4gpu": ["B4"],
    "B4hz": ["B4"],
    "B4feed": ["B4", "B4-g2", "B4-g4"],
    "WIN": [],
    "REDO": [],
}


def tasks(phase):
    out = []
    if phase == "B4":  # first job (bench-b4-20261019-0701): stl + pec arms
        a = 0.5e-3
        for q in (4, 3, 2, 1.5, 1):
            for arm in ("cart-staircase-stl-graded", "cart-conformal-pec-graded"):
                out.append(("B4", arm, a / q))
        out.append(("B4", "cart-conformal-pec-graded-bigdomain", a / 2))
    if phase == "B4b":  # edge-wise staircase ladder and its checks
        a = 0.5e-3
        for q in (4, 3, 2, 1.5, 1):
            out.append(("B4", "cart-staircase-edges-graded", a / q))
        out.append(("B4", "cart-staircase-edges-graded-bigdomain", a / 2))
        out.append(("B4", "cart-staircase-edges-graded-ppw40", a / 2))
    if phase == "WIN":  # DFT-window floor: finest arms at 1x and 3x t_run
        for case in PHASES["B2"] + PHASES["B3"]:
            for wf in (1, 3):
                out.append((case, "tri-uniform", A_NANO / 28, wf))
                out.append((case, "cart-conformal-all-graded", A_NANO / 40, wf))
        return out
    if phase == "REDO":  # B3 (gap stencil metric, sliver-free mesher) + B2 triangles
        out = [t for t in tasks("B3")]
        out += [t for t in tasks("B2") if t[1].startswith("tri")]
        return out
    if phase == "B4gpu":  # fine edge-wise rungs on L4 GPUs (BENCH_GPU=1)
        a = 0.5e-3
        for q in (4, 3, 2):
            out.append(("B4", "cart-staircase-edges-graded", a / q))
        out.append(("B4", "cart-staircase-edges-graded-bigdomain", a / 2))
        out.append(("B4", "cart-staircase-edges-graded-ppw40", a / 2))
    if phase == "B4hz":  # gap/axial cell study at h_t = a/3 (h_z = 0.25, 0.125 mm)
        out = [("B4", "cart-staircase-edges-graded-hz1.5ht", 0.5e-3 / 3),
               ("B4", "cart-staircase-edges-graded-hz0.75ht", 0.5e-3 / 3)]  # fmt: skip
    if phase == "B4feed":  # feed-model and end-cap study at h_t = a/3
        ht = 0.5e-3 / 3
        base = "cart-staircase-edges-graded"
        out = [
            ("B4", f"{base}-sheet", ht),
            ("B4-g2", base, ht),
            ("B4-g4", base, ht),
            ("B4-g4", f"{base}-sheet", ht),
            ("B4", f"{base}-round", ht),
            ("B4", f"{base}-sheet-round", ht),
            ("B4-g2", f"{base}-sheet", ht),
        ]
    if phase.startswith("B4"):
        return out
    names = [n for n, a in arms().items() if not hasattr(a, "hz_ratio")]
    for case in PHASES[phase]:
        ladder = LADDER
        if case == "B3-g2":  # the gap needs h <= 1 nm
            ladder = [h for h in LADDER if h <= 1.0e-9 + 1e-15]
        for h in ladder[::-1]:  # longest first
            for arm in names:
                out.append((case, arm, h))
    return out


def run_tasks(mine, outdir, publish=False, machine=None, phase=None):
    from wakis.benchmarks import get_case, run, save_record

    A = arms()
    Path(outdir).mkdir(parents=True, exist_ok=True)
    for task in mine:
        case_name, arm_name, h = task[:3]
        wf = task[3] if len(task) > 3 else 1
        case = get_case(case_name)
        case.t_run *= wf
        rec = run(case, A[arm_name], h)
        rec["window_factor"] = wf
        tag = f"{case_name}_{arm_name}_{h * 1e9:.3f}nm"  # (B4: h in m * 1e9)
        if wf != 1:
            tag += f"_w{wf:g}"
        if phase == "B3m2":
            tag += "_m2"
            rec["metric_version"] = (
                "m2: stencil mean, second-order gap reconstruction "
                "(tri patch, cart bilinear)"
            )
        if os.environ.get("WAKIS_COMMIT"):
            rec["commit"] = os.environ["WAKIS_COMMIT"]
        save_record(rec, Path(outdir) / f"{tag}.json")
        if publish:
            from wakis.benchmarks.results import publish as pub

            if phase == "B3m2":
                pub_tag = f"{h * 1e9:.4f}nm_m2"
            elif wf != 1:
                pub_tag = f"{h * 1e9:.4f}nm_w{wf:g}"
            else:
                pub_tag = None
            pub(rec, machine=machine, tag=pub_tag)
        m = rec.get("metrics", {})
        msg = rec.get("error") or " ".join(
            f"{k}={m[k]:.4g}"
            for k in ("q_ext_nrms", "q_ext_lam_peak", "enh_peak", "f_res", "r_res")
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
    run_tasks(mine, a.outdir or DATA / a.phase, a.publish, a.machine, phase=a.phase)


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"total {time.time() - t0:.0f} s", flush=True)
