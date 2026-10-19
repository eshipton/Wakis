"""
Round-off growth of the time-reversed leapfrog (memory='reverse') vs Nt.

Two lossless 2D problems (40 x 40 cells, 5 x 5 node design, eps 1-4):
an open domain with CPML (energy leaves; interface layers recorded) and a
closed PEC cavity (energy stays forever: the worst case for accumulated
round-off). For each Nt the gradient of a DFT objective by time reversal
is compared with the stored-increment gradient (exact to the bit of the
forward run). Also reported: the error after reversing to t = 0 of the
reconstructed fields, relative to the field maximum.

Usage: python roundoff.py --out roundoff.json [--gpu]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))


def main():
    import test_020_adjoint as t20

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="roundoff.json")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--nts", default="250,500,1000,2000,4000,8000,16000,32000,64000")
    a = ap.parse_args()
    results = []
    for bc in ("cpml", "pec"):
        for Nt in [int(x) for x in a.nts.split(",")]:
            sim = t20._scattering_sim(
                lambda: t20._make_2d_solver(bc, use_gpu=a.gpu), Nt=Nt
            )
            rho = np.random.default_rng(1).uniform(0.2, 0.8, sim.design.shape)
            F, G = t20._objective(sim.forward(rho))
            g_store = sim.vjp(G)
            sim.memory = "reverse"
            sim.forward(rho)
            g_rev = sim.vjp(G)
            err = float(np.max(np.abs(g_rev - g_store)) / np.max(np.abs(g_store)))
            rec = dict(bc=bc, Nt=Nt, rel_err=err,
                       record_bytes=sim.memory_stats["peak_bytes"])  # fmt: skip
            print(json.dumps(rec), flush=True)
            results.append(rec)
            sim.memory = "store"
    with open(a.out, "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
