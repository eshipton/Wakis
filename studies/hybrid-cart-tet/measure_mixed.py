# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Measure ``mass="consistent"`` vs ``mass="mixed"`` on the test Delaunay mesh,
the N = 16 Delaunay PEC cavity (`cavity_modes`), and the PEC-sphere time-domain
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

from wakis.hybrid import (
    HybridFIT3D,
    box_surface_check,
    delaunay_box_mesh,
    well_centred_tets,
)
from wakis.hybrid.verification import cavity_modes


def _measure_pair(xn, yn, zn, box, tm, extra=None):
    lines = [v[i0 : i1 + 1] for v, (i0, i1) in zip((xn, yn, zn), box)]
    bf_box = box_surface_check(tm, *lines, allow_other=True)
    box_nodes = np.zeros(tm.n_nodes, bool)
    box_nodes[tm.faces[bf_box].ravel()] = True
    box_faces = np.zeros(tm.n_faces, bool)
    box_faces[bf_box] = True
    tet_on_box = np.any(box_nodes[tm.cells], axis=1) | np.any(
        box_faces[tm.cell_faces], axis=1
    )
    wc = well_centred_tets(tm)
    out = dict(extra or {})
    out.update(
        {
            "n_tets": int(tm.n_cells),
            "n_tet_edges": int(tm.n_edges),
            "n_tets_non_well_centred": int((~wc).sum()),
            "n_tets_touching_box": int(tet_on_box.sum()),
        }
    )
    solvers = {}
    for mass in ("consistent", "mixed"):
        s = HybridFIT3D(
            xn,
            yn,
            zn,
            box=box,
            tet_mesh=tm,
            mass=mass,
            mass_solver="chebyshev",
        )
        solvers[mass] = s
        ms = s.msolver
        lumped = int(s.lump_mask.sum()) if mass == "mixed" else 0
        out[mass] = {
            "n_free": int(s.n_free),
            "n_lumped": lumped,
            "n_tet_edges_lumped": lumped,
            "lump_fraction": float(lumped / tm.n_edges),
            "dt": float(s.dt),
            "dt_max": float(s.dt_max),
            "degree": int(ms.degree),
            "lam_lo": float(ms.lam_lo),
            "lam_hi": float(ms.lam_hi),
            "coupled_block": int(ms.S.size),
        }
    out["n_tet_edges_lumped"] = out["mixed"]["n_lumped"]
    out["lump_fraction"] = out["mixed"]["lump_fraction"]
    return out, solvers


def measure_test_delaunay():
    h = 0.1
    xn = np.arange(0, 1.0 + 1e-9, h)
    yn = np.arange(0, 0.8 + 1e-9, h)
    zn = np.arange(0, 0.6 + 1e-9, h)
    box = ((3, 7), (2, 6), (2, 4))
    lines = (xn[3:8], yn[2:7], zn[2:5])
    tm = delaunay_box_mesh(*lines, h=0.06, r=2, seed=0)
    out, _ = _measure_pair(xn, yn, zn, box, tm)
    return out


def measure_cavity_n16():
    f_cons, f_ex, s_cons = cavity_modes(16, patch="delaunay", n_modes=3)
    box = tuple(zip(s_cons.box_lo, s_cons.box_hi))
    out, solvers = _measure_pair(*s_cons.block.nodes, box, s_cons.tet)
    f_mix = solvers["mixed"].eigenfrequencies(3, 0.9 * f_ex[0], 1.15 * f_ex[-1])
    diff_mc = float(abs(f_mix[0] - f_cons[0]))
    diff_ca = float(abs(f_cons[0] - f_ex[0]))
    out.update(
        {
            "f_analytic_1": float(f_ex[0]),
            "f_consistent_1": float(f_cons[0]),
            "f_mixed_1": float(f_mix[0]),
            "diff_mixed_consistent": diff_mc,
            "diff_consistent_analytic": diff_ca,
            "ratio": float(diff_mc / diff_ca),
        }
    )
    return out


def measure_case(N):
    h = 1.0 / N
    a = 2.0 / (2 * np.pi)
    nb = int(np.ceil((a + max(0.12, 2.2 * h)) / h))
    nh = nb + 3
    v = h * np.arange(-nh, nh + 1)
    i0, i1 = nh - nb, nh + nb
    box = ((i0, i1),) * 3
    tm = delaunay_box_mesh(
        v[i0 : i1 + 1],
        v[i0 : i1 + 1],
        v[i0 : i1 + 1],
        h=h,
        sphere=((0, 0, 0), a),
        hole=True,
    )
    out, _ = _measure_pair(v, v, v, box, tm, extra={"N": int(N)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", type=int, nargs="*", default=[10, 16])
    args = ap.parse_args()
    res = {
        "test_delaunay": measure_test_delaunay(),
        "cavity_N16": measure_cavity_n16(),
    }
    for N in args.ns:
        res[f"N={N}"] = measure_case(N)
    print(json.dumps(res, indent=2), flush=True)


if __name__ == "__main__":
    main()
