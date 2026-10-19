"""
Long-run energy of the LTS leapfrog in a closed PEC cavity (rod eps 4, radius
0.5 in a cavity of radius 1, graded mesh h = 0.08 -> 0.16), TM, random
initial field, 1e5 coarse steps. Variants: correct LTS (nu = 0.15), the
unstabilized Diaz-Grote polynomial (nu = 0) and two mutations (no coupling of
F to its coarse neighbours; polynomial normalization off by 1 %).

Writes data/energy.json (tiny: run locally under bin/memguard).
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "tests"))
from test_047_unstructured_lts import graded_rod_mesh  # noqa: E402

from wakis.unstructured import UnstructuredFIT2D  # noqa: E402
from wakis.unstructured import lts as L  # noqa: E402

N_STEPS, EVERY = 100_000, 250


def run(mesh, variant):
    kw = dict(p=2)
    if variant == "nu=0":
        kw.update(nu=0.0, rho_max=1.0)
    s = UnstructuredFIT2D(mesh, {0: [4.0, 1.0]}, lts=kw)
    if variant == "global":
        s = UnstructuredFIT2D(mesh, {0: [4.0, 1.0]})
    if variant == "mutant: buffer":
        s.lts.kappa[:] = 0
    if variant == "mutant: normalization":
        s.lts.out *= 1.01
    s.u = np.random.default_rng(0).standard_normal(s.free.size)
    t, E, P = [], [], []
    for n in range(N_STEPS):
        u0 = s.u.copy()
        s.one_step()
        if n % EVERY == 0:
            t.append(s.n * s.dt)
            E.append(L.modified_energy(s, u0, s.u))
            P.append(s.energy())
    return dict(t=t, E=E, P=P, dt=s.dt, rho=float(s.courant))


def main():
    mesh = graded_rod_mesh(0.08)
    out = {}
    for v in ("global", "LTS nu=0.15", "nu=0", "mutant: buffer",
              "mutant: normalization"):  # fmt: skip
        out[v] = run(mesh, v)
        E = np.array(out[v]["E"])
        print(v, "rho %.4f" % out[v]["rho"], "drift %.2e" % np.abs(E / E[0] - 1).max())
    (HERE / "data").mkdir(exist_ok=True)
    (HERE / "data" / "energy.json").write_text(json.dumps(out))


if __name__ == "__main__":
    main()
