"""
Lossless Drude rod (radius b = a/2, k_p a = 4) in a circular PEC cavity
(radius a), TE (E in the plane, crossing the metal surface): the n = 1
surface-plasmon mode (k a = 3.64691, eps(w) = -0.20) by time-domain ring-down
and harmonic inversion, against the exact Bessel / modified-Bessel matching
(`wakis.unstructured.analytic.drude_rod_in_cavity_te`).

Usage: python rod2d.py <n cells across 2.2 a> <conformal: pec|all|tensor|False>
       [out.json]
The leapfrog error is removed exactly for a lossless Drude medium:
the discrete scheme is the continuous one at 2 sin(w dt / 2) / dt.
"""

import json
import sys
import time

import numpy as np
import pyvista as pv
from scipy.constants import c

import wakis
from wakis.dispersive import DispersiveMaterial, Drude
from wakis.resonances import harminv
from wakis.sources import CurrentSource, GaussianSource
from wakis.unstructured import analytic

A, B, KP = 1.0, 0.5, 4.0
R_SRC = 0.62  # source/probe radius (outside the rod)
CENTER = np.array([0.0031, -0.0017])  # off the grid nodes


def reference():
    return analytic.drude_rod_in_cavity_te(A, B, KP, n=1)[1]


def run(n, conformal, periods=300, gamma=0.0):
    t0 = time.time()
    k_sp = reference()
    f_sp = k_sp * c / (2 * np.pi)
    h = 2.2 * A / n
    lo, hi = CENTER - 1.1 * A, CENTER + 1.1 * A

    def cyl(r):
        return pv.Cylinder(center=(*CENTER, 0), direction=(0, 0, 1), radius=r,
                           height=20 * h, resolution=4 * 1440).triangulate()  # fmt: skip

    drude = DispersiveMaterial(1.0, [Drude(KP * c, gamma * KP * c)], name="drude")
    grid = wakis.GridFIT3D(
        lo[0], hi[0], lo[1], hi[1], -h / 2, h / 2, n, n, 1,
        stl_solids={"cavity": cyl(A), "rod": cyl(B)},
        stl_materials={"cavity": "vacuum", "rod": drude},
        conformal="tensor" if conformal == "coupled" else conformal, verbose=0,
    )  # fmt: skip
    solver = wakis.SolverFIT3D(
        grid, bc_low=["pec", "pec", "periodic"], bc_high=["pec", "pec", "periodic"],
        bg="pec", use_stl=True, verbose=0,
    )  # fmt: skip
    if conformal == "coupled":  # research: non-passive dispersive tensor
        from coupled import make_coupled

        make_coupled(solver)
    pulse = GaussianSource(frequency=f_sp, fwidth=0.3 * f_sp)
    # x-dipole filter: E_x sources and probes at (+-r, 0) (weight 1) and
    # (0, +-r) (weight -1/2), the near field of an x-dipole outside the rod;
    # modes odd in x or y (n = 0, 2, ...) cancel up to the grid asymmetry
    pts = [
        ([R_SRC, 0], 1.0),
        ([-R_SRC, 0], 1.0),
        ([0, R_SRC], -0.5),
        ([0, -R_SRC], -0.5),
    ]
    srcs = [
        CurrentSource(pulse, component="x", center=(*(CENTER + p), 0.0),
                      amplitude=w)
        for p, w in pts
    ]  # fmt: skip

    class Probe:
        def __init__(self, point, comp):
            coords = [solver.grid.x[:-1], solver.grid.y[:-1], solver.grid.z[:-1]]
            a = "xyz".index(comp)
            d = [solver.grid.dx, solver.grid.dy, solver.grid.dz][a]
            coords[a] = coords[a] + d / 2
            self.index = (
                *(int(np.argmin(abs(q - p))) for q, p in zip(coords, point)),
                comp,
            )
            self.values = []

        def update(self, solver, t):
            self.values.append(float(solver.E[self.index]))

    probes = [Probe((*(CENTER + p), 0.0), "x") for p, _ in pts]
    Nt = int((pulse.end_time + periods / f_sp) / solver.dt)
    solver.emsolve(Nt, source=srcs, monitors=probes)
    start = int(np.ceil(pulse.end_time / solver.dt))
    sig = sum(w * np.array(p.values[start:]) for p, (_, w) in zip(probes, pts))
    step = max(1, int(1 / (8 * f_sp * solver.dt)))
    res = harminv(sig[::step], step * solver.dt, 0.85 * f_sp, 1.15 * f_sp,
                  t0=(start + 1) * solver.dt)  # fmt: skip
    amp = np.abs(res.amplitude)
    i = np.argmax(amp)
    f = 2 / solver.dt * np.sin(np.pi * res.frequency * solver.dt) / (2 * np.pi)
    out = [dict(f=float(f[i]), decay=float(res.decay_rate[i]),
                modes=(f / f_sp - 1).tolist(), amps=(amp / amp[i]).tolist())]  # fmt: skip
    errs = [o["f"] / f_sp - 1 for o in out]
    cells = np.pi * A**2 / h**2
    return dict(n=n, conformal=str(conformal), cells=float(cells), dt=solver.dt,
                Nt=Nt, err=errs, probes=out, f_ref=f_sp,
                time=time.time() - t0)  # fmt: skip


if __name__ == "__main__":
    n = int(sys.argv[1])
    conf = sys.argv[2]
    conf = False if conf == "False" else conf
    r = run(n, conf)
    print(json.dumps(r))
    if len(sys.argv) > 3:
        with open(sys.argv[3], "a") as fh:
            fh.write(json.dumps(r) + "\n")
