"""Diagnostic: wall time of the LTS setup stages on the study meshes (Batch)."""
import sys
import time

import numpy as np

sys.path.insert(0, "studies/subgrid3d")
import run_study_lts as R  # noqa: E402

from wakis.subgrid3d.lts import SubgridLTS, is_stable, level_margins, stability_ratio  # noqa: E402

for h, r in ((0.1, 3), (0.05, 2), (0.05, 3), (0.1 / 3, 2)):
    t = time.perf_counter()
    s = R.make_solver("big", "global", h, r)
    print(h, r, "global setup", time.perf_counter() - t, s.n_free, flush=True)
    for f in (0.9, 0.99, 1.02):
        dt = f * r * s.dt_max
        t = time.perf_counter()
        L = SubgridLTS(s, dt)
        tb = time.perf_counter() - t
        t = time.perf_counter()
        rho = stability_ratio(L)
        tr = time.perf_counter() - t
        t = time.perf_counter()
        m = level_margins(L)
        tm = time.perf_counter() - t
        print(f"  f={f} build {tb:.1f}s rho={rho:.5f} {tr:.1f}s margins={m} {tm:.1f}s", flush=True)
