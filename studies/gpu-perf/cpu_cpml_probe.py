"""Where does the CPU CPML step spend its time, and what would merging the
curl products save? (Report only: the CPU path is not changed.)

  OMP_NUM_THREADS=1 python cpu_cpml_probe.py [--n 64]

Times (single core, after one warm-up step):
  step        SolverFIT3D._one_step_cpml (CPU, SciPy)
  spmv12      the 12 per-component derivative products alone
  merged2     two 3N-row products with the merged curl rows (A - B),
              i.e. the cost of the curl if it were formed like the plain step
  face_rows   the derivative terms restricted to the CPML face rows (needed
              for psi), as row-sliced operators
  plain       SolverFIT3D._one_step on the same grid without CPML
"""

import argparse
import cProfile
import io
import json
import pstats
import sys
import pathlib
import time

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import cases  # noqa: E402

from wakis.gpu_fused import _E_TERMS, _H_TERMS, _block  # noqa: E402


def timeit(fn, reps=10):
    fn()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return 1e3 * (time.perf_counter() - t0) / reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64)
    a = ap.parse_args()
    s, srcs, _ = cases.cpml3d(False, None, n=a.n)
    srcs[0].update(s, 0.0)
    s.one_step()
    E, H = s.E.array, s.H.array
    out = {"cells": int(s.N), "step": timeit(s.one_step)}
    ops = [t[4] for t in _H_TERMS + _E_TERMS]

    def spmv12():
        for op in ops:
            getattr(s, op) @ E[: s.N]

    out["spmv12"] = timeit(spmv12)
    mats = {o: sp.csr_matrix(getattr(s, o)) for o in ops}
    MH = (_block(mats, _H_TERMS, 1.0) - _block(mats, _H_TERMS, -1.0)).tocsr()
    ME = (_block(mats, _E_TERMS, 1.0) - _block(mats, _E_TERMS, -1.0)).tocsr()
    out["merged2"] = timeit(lambda: (MH @ E, ME @ H))
    rows = []
    for T, d, S, sg, op, prefix in _H_TERMS + _E_TERMS:
        for side in ("low", "high"):
            if hasattr(s, f"idx_{d}_{side}"):
                idx = np.asarray(getattr(s, f"idx_{d}_{side}"))
                rows.append((mats[op][idx], "xyz".index(S)))
    out["face_rows"] = timeit(lambda: [R @ E[k * s.N : (k + 1) * s.N] for R, k in rows])
    sp_, srcs2, _ = cases.plain3d(False, None, n=a.n)
    srcs2[0].update(sp_, 0.0)
    sp_.one_step()
    out["plain"] = timeit(sp_.one_step)
    prof = cProfile.Profile()
    prof.enable()
    for _ in range(5):
        s.one_step()
    prof.disable()
    buf = io.StringIO()
    pstats.Stats(prof, stream=buf).sort_stats("tottime").print_stats(8)
    out["profile_tottime"] = buf.getvalue()[-2500:]
    print(
        json.dumps({k: v for k, v in out.items() if k != "profile_tottime"}, indent=1)
    )
    print(out["profile_tottime"])


if __name__ == "__main__":
    main()
