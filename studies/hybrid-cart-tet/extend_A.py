"""Part A extension: TE10 |R| for N = 20, 24, 28 (Kuhn r1, Delaunay r1, floor), packed on all cores.
Writes results_A_ext.json (same format as run_study.py part A)."""

import json
import os
import sys
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_study import HERE, task_A  # noqa: E402

if __name__ == "__main__":
    tasks = [(N, k) for N in (28, 24, 20) for k in ("kuhn_r1", "delaunay_r1", "floor")]
    with Pool(len(tasks)) as p:
        res = p.map(task_A, tasks, chunksize=1)
    json.dump(
        {"meta": {"script": "extend_A.py"}, "results": res},
        open(os.path.join(HERE, "results_A_ext.json"), "w"),
        indent=1,
    )
    for r in res:
        print(r, flush=True)
