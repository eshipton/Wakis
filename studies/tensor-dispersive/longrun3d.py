"""
Long-run stability in 3D: lossless Drude sphere (eps < 0 at the plasmon) in
a PEC sphere with conformal='all', 1e5 steps after a pulse at the automatic
time step. Prints the E norm every 1e4 steps (bounded = stable).

Usage: python longrun3d.py <n cells across 2b> [drude|gold] [all|pec] [steps/10] [--gpu]
"""

import sys

import numpy as np
import pyvista as pv
from scipy.constants import c

import wakis
from wakis.dispersive import DispersiveMaterial, Drude, dispersive_lib
from wakis.sources import CurrentSource, GaussianSource

n = int(sys.argv[1])
kind = sys.argv[2] if len(sys.argv) > 2 else "drude"
conformal = sys.argv[3] if len(sys.argv) > 3 else "all"
chunk = int(sys.argv[4]) if len(sys.argv) > 4 else 10000
use_gpu = "--gpu" in sys.argv
center = np.array([0.0071, -0.0043, 0.0029])
if kind == "gold":  # 100 nm sphere radius, Rakic gold
    a, b, mat = 200e-9, 100e-9, dispersive_lib["gold"]
    f0 = c / 520e-9
else:
    a, b, mat = 1.0, 0.5, DispersiveMaterial(1.0, [Drude(6 * c, 0.0)])
    f0 = 4.53 * c / (2 * np.pi)
center = center * a
h = 2 * b / n
m = int(np.ceil(2.2 * a / h))
lo, hi = center - m * h / 2, center + m * h / 2


def sphere(r):
    return pv.Sphere(radius=r, center=center, theta_resolution=300, phi_resolution=300)


grid = wakis.GridFIT3D(lo[0], hi[0], lo[1], hi[1], lo[2], hi[2], m, m, m,
                       stl_solids={"cavity": sphere(a), "core": sphere(b)},
                       stl_materials={"cavity": "vacuum", "core": mat},
                       conformal=conformal, verbose=0)  # fmt: skip
solver = wakis.SolverFIT3D(grid, bc_low=["pec"] * 3, bc_high=["pec"] * 3, bg="pec",
                           use_stl=True, use_gpu=use_gpu, verbose=0)  # fmt: skip
net = getattr(solver, "dispersive_network", None)
print("cells", m**3, "network edges", net and net.rows.size,
      "bound", net and net.stability_bound(solver.conformal_dt_max))  # fmt: skip
pulse = GaussianSource(frequency=f0, fwidth=0.5 * f0)
src = CurrentSource(pulse, component="z", center=tuple(center + [0, 0, 1.3 * b]))
solver.emsolve(int(pulse.end_time / solver.dt) + 1, source=[src])
for k in range(10):
    solver.emsolve(chunk)
    E = solver.E.toarray()
    E = E.get() if hasattr(E, "get") else E
    print(k, "%.4e" % np.linalg.norm(E), flush=True)
