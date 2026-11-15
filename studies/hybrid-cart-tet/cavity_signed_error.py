# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Signed lowest-mode resonance frequency errors of `mass="consistent"` and
`mass="mixed"` on the Delaunay PEC cavity (`cavity_modes(N, patch="delaunay",
n_modes=3)`) for `N in {12, 16, 20}`.

Prints one JSON object to stdout and writes no files.
"""

import argparse
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from wakis.hybrid import HybridFIT3D
from wakis.hybrid.verification import cavity_modes


def run_cavity_signed_error(ns=(12, 16, 20)):
    out = {}
    for N in ns:
        N = int(N)
        f_cons, f_ex, s_cons = cavity_modes(N, patch="delaunay", n_modes=3)
        box = tuple(zip(s_cons.box_lo, s_cons.box_hi))
        s_mix = HybridFIT3D(
            *s_cons.block.nodes,
            box=box,
            tet_mesh=s_cons.tet,
            mass="mixed",
            dt=1.0,
        )
        f_mix = s_mix.eigenfrequencies(3, 0.9 * f_ex[0], 1.15 * f_ex[-1])
        f_an1 = float(f_ex[0])
        f_c1 = float(f_cons[0])
        f_m1 = float(f_mix[0])
        out[f"N={N}"] = {
            "N": N,
            "f_analytic_1": f_an1,
            "f_consistent_1": f_c1,
            "f_mixed_1": f_m1,
            "signed_rel_err_consistent": float((f_c1 - f_an1) / f_an1),
            "signed_rel_err_mixed": float((f_m1 - f_an1) / f_an1),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", type=int, nargs="*", default=[12, 16, 20])
    args = ap.parse_args()
    out = run_cavity_signed_error(ns=args.ns)
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
