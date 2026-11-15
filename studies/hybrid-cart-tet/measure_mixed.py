# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Measure ``mass="consistent"`` vs ``mass="mixed"`` on the PEC-sphere time-domain
geometry of `studies/hybrid-cart-tet/run_study.py` part D at N = 10 and N = 16.

Prints one JSON object to stdout and writes no files.
"""

import argparse
import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

from wakis.hybrid import HybridFIT3D, delaunay_box_mesh


def measure_case(N):
    h = 1.0 / N
    a = 2.0 / (2 * np.pi)
    nb = int(np.ceil((a + max(0.12, 2.2 * h)) / h))
    nh = nb + 3
    v = h * np.arange(-nh, nh + 1)
    i0, i1 = nh - nb, nh + nb
    tm = delaunay_box_mesh(
        v[i0 : i1 + 1],
        v[i0 : i1 + 1],
        v[i0 : i1 + 1],
        h=h,
        sphere=((0, 0, 0), a),
        hole=True,
    )
    out = {"N": int(N), "n_tets": int(tm.n_cells), "n_tet_edges": int(tm.n_edges)}
    for mass in ("consistent", "mixed"):
        s = HybridFIT3D(
            v,
            v,
            v,
            box=((i0, i1),) * 3,
            tet_mesh=tm,
            mass=mass,
            mass_solver="chebyshev",
        )
        ms = s.msolver
        lumped = int(s.lump_mask.sum()) if mass == "mixed" else 0
        out[mass] = {
            "n_free": int(s.n_free),
            "n_lumped": lumped,
            "lump_fraction": float(lumped / tm.n_edges),
            "dt": float(s.dt),
            "dt_max": float(s.dt_max),
            "degree": int(ms.degree),
            "lam_lo": float(ms.lam_lo),
            "lam_hi": float(ms.lam_hi),
            "coupled_block": int(ms.S.size),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", type=int, nargs="+", default=[10, 16])
    args = ap.parse_args()
    res = {f"N={N}": measure_case(N) for N in args.ns}
    print(json.dumps(res, indent=2), flush=True)


if __name__ == "__main__":
    main()
