"""
DFT-window convergence of the shadow-focus objective: J/J0 of the bare
cylinder and of the binarized design (eps per triangle from fields.npz) for
k * Nt steps, k = 1, 2, 4, 8 (one process per run). Writes window.json.

    python studies/unstructured-adjoint/check_window.py
"""

import json
import os
from multiprocessing import Pool

import numpy as np

from shadow_focus import F0, FOCUS, HERE, setup
from wakis.unstructured.adjoint import CellDFTMonitor, CellMaterialGrid, UnstructuredAdjointSimulation

CPW, PERIODS = 20, 60


def run(args):
    k, which = args
    mesh, make, dt, Nt, cells, _, _ = setup(CPW, PERIODS)
    eps = np.load(os.path.join(HERE, "fields.npz"))["eps_bin"] if which == "design" else np.ones(cells.size)
    sim = UnstructuredAdjointSimulation(make, CellMaterialGrid(mesh, cells, 0.0, 1.0),
                                        {"f": CellDFTMonitor(F0, [FOCUS])}, k * Nt)  # fmt: skip
    X = sim.forward(eps, record=False)["f"]["dft"][0, 0]
    return k, which, float(abs(X) ** 2), Nt


if __name__ == "__main__":
    jobs = [(k, w) for k in (1, 2, 4, 8) for w in ("bare", "design")]
    with Pool(4) as p:
        out = p.map(run, jobs)
    J = {(k, w): v for k, w, v, _ in out}
    res = {f"{k}Nt": dict(J0=J[k, "bare"], J_over_J0=J[k, "design"] / J[k, "bare"])
           for k in (1, 2, 4, 8)}  # fmt: skip
    res["Nt"] = out[0][3]
    print(json.dumps(res, indent=1))
    json.dump(res, open(os.path.join(HERE, "window.json"), "w"), indent=1)
