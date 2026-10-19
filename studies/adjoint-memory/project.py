"""
Memory ladder projected to the full ceviche WDM / bend, the invrs 2D
metalens and a 3D metalens-sized problem.

Inputs per problem: cells, Nt, design edges (from `invrs_wakis.resources`
geometry, printed by `sizes.py` in this folder) and the snapshot / interface
record sizes, which are counted exactly on a real solver of the same grid
(2D problems) or on a small 3D solver scaled by interface area (3D).

Per rung: bytes held for the gradient and extra forward steps:
- store (float64, float32): Nt x design edges x itemsize, 0 extra steps;
- checkpoint: `plan_checkpoints` for budgets from 0.1 GB to the store size
  (exact recompute count of the binomial schedule);
- reverse: (interface layers + source supports) x Nt x 8 + 3 field states,
  0 extra forward steps (one reversed step per adjoint step instead, ~ the
  cost of a forward step);
- nyquist (approximate): store / decimation, decimation = 1 / (2 f_max dt)
  / 2 (factor 2 margin below Nyquist).

Usage (VM; the WDM solver takes ~1 GB): python project.py --out projection.json
"""

import argparse
import json

import numpy as np

import wakis
from wakis.inverse import memory as mem


def counts(nx, ny, nz, n_pml, xy_only=False):
    """Snapshot bytes and interface-layer edges of a CPML solver."""
    dx = 1e-3
    grid = wakis.GridFIT3D(0, nx * dx, 0, ny * dx, 0, nz * dx, nx, ny, nz, verbose=0)
    bc = ["cpml", "cpml", "periodic" if xy_only else "cpml"]
    solver = wakis.SolverFIT3D(grid, bc_low=bc, bc_high=bc, n_pml=n_pml, verbose=0)
    A, B = mem._curl_operators(solver)
    inside = mem._interior_mask(solver)
    I = np.flatnonzero(inside)
    g_H = np.setdiff1d(np.unique(B[I].indices), I).size
    g_E = np.setdiff1d(np.unique(A[I].indices), I).size
    psi = sum(getattr(solver, k).nbytes for k in vars(solver) if k.startswith("psi"))
    state = 3 * solver.E.array.nbytes + psi  # E, H, J
    return dict(state_bytes=int(state), layer_edges=int(g_H + g_E))


PROBLEMS = {
    # name: (cells (nx, ny, nz), Nt, design edges, f_max [Hz], dt [s], n_pml)
    "ceviche_waveguide_bend": ((313, 313, 1), 56587, 25600, 2.37e14, 2.2407e-17, 20),
    "ceviche_wdm": ((825, 761, 1), 56587, 409600, 2.37e14, 2.2407e-17, 20),
}
# 3D metalens-sized case (assumption, see README): 10 x 10 lambda aperture,
# 20 cells / lambda, 4 lambda tall, 10-cell CPML: 220 x 220 x 100 cells;
# design: 10-cell pillar layer, 200 x 200 x 10 cells x 3 E components;
# Nt = 4000; f_max = 1.2 f0 at cfl 0.5 -> 1 / (2 f_max dt) ~ 41.
METALENS_3D = ((220, 220, 100), 4000, 3 * 200 * 200 * 10, 1.2, 1 / 41 / 2 / 1.2)


def ladder(cells, Nt, design_edges, state_bytes, layer_edges, source_edges,
           decimation):  # fmt: skip
    row64 = 8 * design_edges
    out = {"store_f64": dict(bytes=Nt * row64, extra=0),
           "store_f32": dict(bytes=Nt * row64 // 2, extra=0)}  # fmt: skip
    curve = []
    for budget in np.geomspace(max(2 * state_bytes, 1e8), Nt * row64, 25):
        try:
            p = mem.plan_checkpoints(Nt, state_bytes, row64, budget)
        except MemoryError:
            continue
        curve.append(dict(budget=float(budget), bytes=int(p["bytes"]),
                          slots=p["slots"], block=p["block"],
                          extra=int(p["recompute_steps"])))  # fmt: skip
    out["checkpoint_curve"] = curve
    rec = (layer_edges + source_edges) * Nt * 8 + 3 * (2 * state_bytes // 3)
    out["reverse"] = dict(bytes=int(rec), extra=0, record_edges=layer_edges)
    out["nyquist"] = dict(bytes=Nt * row64 // decimation, extra=0,
                          decimation=decimation)  # fmt: skip
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="projection.json")
    a = ap.parse_args()
    res = {}
    for name, (cells, Nt, de, fmax, dt, n_pml) in PROBLEMS.items():
        c = counts(*cells, n_pml, xy_only=True)
        dec = max(1, int(1 / (2 * fmax * dt) / 2))
        res[name] = dict(cells=cells, Nt=Nt, design_edges=de, **c,
                         ladder=ladder(cells, Nt, de, c["state_bytes"],
                                       c["layer_edges"], 200, dec))  # fmt: skip
        print(name, json.dumps({k: v for k, v in res[name].items() if k != "ladder"}))
    # 3D: count on a 60 x 60 x 40 solver and scale state by cells, layers by area
    cells, Nt, de, _, _ = METALENS_3D
    small = (60, 60, 40)
    c = counts(*small, 10)
    n_small, n_big = np.prod(small), np.prod(cells)

    def area(s, p=10):
        x, y, z = (v - 2 * p for v in s)
        return 2 * (x * y + y * z + z * x)

    state = int(c["state_bytes"] * n_big / n_small)
    layers = int(c["layer_edges"] * area(cells) / area(small))
    tfsf = 6 * 200 * 200 * 2  # TF/SF box faces around the lens (J and H rows)
    res["metalens_3d"] = dict(cells=cells, Nt=Nt, design_edges=de, state_bytes=state,
                              layer_edges=layers,
                              ladder=ladder(cells, Nt, de, state, layers, tfsf, 41))
    print("metalens_3d", state / 1e9, "GB/snapshot", layers, "layer edges")
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1, default=int)


if __name__ == "__main__":
    main()
