"""
Compare `wakis.estimate` with measured runs of `benchmarks/calibrate_memory.py`.

Prints, per run, the measured and estimated peak host RAM (without the safety
factor), GPU memory, time per step and setup time, and the largest relative
errors over the held-out sizes (those not used to fit the coefficients)::

    python benchmarks/validate_estimate.py cal_plain.json cal_conf.json ...
"""

import json
import sys

from wakis.estimate import estimate, estimate_unstructured

# Grid sizes N (N^3 cells) used for the fits; all others are held out
FIT_N = {32, 64, 96, 128, 160}
FIT_N_UNSTRUCTURED = {8, 12, 16, 20}
FIT_N_CONFORMAL_GPU = {64, 128}


def predict(r):
    if r["suite"] == "unstructured":
        return estimate_unstructured(r["cells"], safety=1.0)
    kw = dict(
        dtype=r["dtype"],
        backend="gpu" if r["gpu"] else "cpu",
        cpml=r["bc"] == "cpml",
        safety=1.0,
    )
    if r["suite"] == "conformal":
        kw.update(conformal=r["mode"], interface_cells=r["interface_cells"])
    if r["suite"] == "dispersive":
        kw.update(
            dispersive_edges=r["dispersive_edges"], n_poles=int(r["mode"].split(":")[1])
        )
    return estimate(r["cells"], **kw)


def main():
    rows = [r for name in sys.argv[1:] for r in json.load(open(name))]
    worst = {}
    for r in rows:
        est = predict(r)
        fit = FIT_N_UNSTRUCTURED if r["suite"] == "unstructured" else FIT_N
        if r["suite"] == "conformal" and r["gpu"]:
            fit = FIT_N_CONFORMAL_GPU
        held = r["N"] not in fit
        errs = {
            "host": est.host_bytes / r["peak"] - 1,
            "step": est.step_seconds / r["step_s"] - 1,
        }
        if r.get("gpu"):
            errs["gpu"] = est.gpu_bytes / r["gpu_bytes"] - 1
        if r["suite"] != "unstructured":
            errs["setup"] = est.setup_seconds / r["setup_s"] - 1
        tag = f"{r['suite']}/{r.get('mode', '-')}/{'gpu' if r['gpu'] else 'cpu'}/{r['bc']}/{r['dtype']}"
        print(f"{'held' if held else 'fit '} {tag:38s} N={r['N']:4d} "
              + " ".join(f"{k} {v:+6.1%}" for k, v in errs.items()))  # fmt: skip
        if held:
            for k, v in errs.items():
                key = (r["suite"], k)
                worst[key] = max(worst.get(key, 0.0), abs(v))
    print("\nlargest held-out errors:")
    for (suite, k), v in sorted(worst.items()):
        print(f"  {suite:13s} {k:6s} {v:6.1%}")


if __name__ == "__main__":
    main()
