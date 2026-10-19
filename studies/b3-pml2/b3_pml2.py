"""
B3 gap-enhancement plateau of graded triangles: boundary reflection or gap
reconstruction?

Phase ``diag``: B3-g5, h = 1.0 / 0.71 / 0.5 / 0.35 nm, absorber variants
(base: 10 PML rings of h_out from 110 nm; 20 rings; PML from 200 nm; outer
mesh ppw 40; 20 rings at h_out / 2). Every run also reports the gap
enhancement from a second-order patch reconstruction (``metrics_patch``)
next to the suite's pointwise Raviart-Thomas one (``metrics``).

Phase ``final``: the chosen configuration on B3-g2/g5/g10 over the suite
ladder, published as ``tri-graded-pml2``.

    python b3_pml2.py list --phase diag
    python b3_pml2.py run --phase diag --shard I --nshards N --outdir DIR
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

NM = 1e-9
DIAG_H = [1.0 * NM, 1e-9 / 1.4, 0.5 * NM, 0.5e-9 / 1.4]
VARIANTS = {
    "base": dict(),
    "pml20": dict(n_pml=20),
    "r200": dict(r_pml=200 * NM),
    "ppw40": dict(ppw_out=40),
    "pml20-half": dict(n_pml=20, pml_dr=0.5),
}
FINAL = dict()  # filled after the diagnostic phase
LADDER = [20 * NM / q for q in (5, 7, 10, 14, 20, 28, 40)]


VARIANTS2 = {
    "n10": dict(n_pml=10),
    "n20": dict(n_pml=20),
    "n40": dict(n_pml=40),
    "n80": dict(n_pml=80),
    "d80n20": dict(n_pml=20, pml_d=80 * NM),
    "d120n30": dict(n_pml=30, pml_d=120 * NM),
}
VARIANTS.update(VARIANTS2)


def tasks(phase):
    if phase == "diag2":  # boundary influence on enh: |enh(variant) - enh(n80)|
        return [("B3-g5", v, h) for h in (1e-9 / 1.4, 0.5 * NM) for v in VARIANTS2]
    if phase == "diag":
        return [
            ("B3-g5", v, h)
            for h in DIAG_H
            for v in ("base", "pml20", "r200", "ppw40", "pml20-half")
        ]
    out = []
    for case in ("B3-g5", "B3-g10", "B3-g2"):
        lad = LADDER if case != "B3-g2" else [h for h in LADDER if h <= 1.0e-9 + 1e-15]
        out += [(case, "final", h) for h in lad[::-1]]
    return out


def arm_for(variant):
    from wakis.benchmarks.arms.triangles2d import Triangles2DArm

    kw = dict(FINAL) if variant == "final" else dict(VARIANTS[variant])
    return Triangles2DArm(gap_patch=True, name="tri-graded-pml2", **kw)


def run_task(case_name, variant, h):
    from wakis.benchmarks import get_case
    from wakis.benchmarks.runner import _plain

    case = get_case(case_name)
    arm = arm_for(variant)
    rec = dict(case=case_name, arm=arm.name, variant=variant, resolution=h,
               config={k: getattr(arm, k) for k in ("n_pml", "pml_dr", "pml_d", "r_pml", "ppw_out", "pml", "ratio")})  # fmt: skip
    t0 = time.time()
    try:
        res = arm.run(case, h)
        rec["cost"] = _plain(res["cost"])
        rec["metrics"] = _plain(case.evaluate(res))
        rec["metrics_patch"] = _plain(
            case.evaluate(dict(res, e_gap=res["e_gap_patch"]))
        )
    except Exception as e:  # noqa: BLE001
        import traceback

        rec["error"], rec["traceback"] = repr(e), traceback.format_exc()
    rec["wall_s"] = time.time() - t0
    try:
        with open("/proc/cpuinfo") as f:
            rec["cpu"] = next(x.split(":")[1].strip() for x in f if "model name" in x)
    except OSError:
        pass
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["list", "run"])
    ap.add_argument("--phase", default="diag")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--outdir", default="out")
    ap.add_argument("--final", default=None, help="JSON arm kwargs for the final phase")
    a = ap.parse_args()
    if a.final:
        FINAL.update(json.loads(a.final))
    tl = tasks(a.phase)
    if a.mode == "list":
        for i, t in enumerate(tl):
            print(i, *t)
        return
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    Path(a.outdir).mkdir(parents=True, exist_ok=True)
    for case_name, variant, h in tl[a.shard :: a.nshards]:
        rec = run_task(case_name, variant, h)
        tag = f"{case_name}_{variant}_{h * 1e9:.3f}nm"
        (Path(a.outdir) / f"{tag}.json").write_text(json.dumps(rec))
        m, mp = rec.get("metrics", {}), rec.get("metrics_patch", {})
        print(tag, rec.get("error") or f"enh_peak RT {m.get('enh_peak'):.4g} patch {mp.get('enh_peak'):.4g} "
              f"qext {m.get('q_ext_nrms'):.3g} refl {m.get('boundary_reflection', float('nan')):.2e}", flush=True)  # fmt: skip


if __name__ == "__main__":
    main()
