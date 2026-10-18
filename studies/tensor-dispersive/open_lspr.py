"""
Dipole LSPR of an isolated lossless Drude sphere (b = 0.2 m, k_p = 4 1/m) in
vacuum with CPML, as in the user-guide example: one z-directed source and
probe on the axis, harmonic inversion of the ring-down, leapfrog error
removed. Reference: complex Mie a_1 pole k = 2.1436 + 0.0549i 1/m
(mie.mie_a1_pole); quasi-static k_p / sqrt(3) = 2.309 1/m.

Usage: python open_lspr.py <n cells across 2b> <conformal> <half width [m]> [out.jsonl] [--gpu]
"""

import json
import sys
import time

import mie
import numpy as np
import pyvista as pv
from scipy.constants import c

import wakis
from wakis.dispersive import DispersiveMaterial, Drude
from wakis.resonances import harminv
from wakis.sources import CurrentSource, GaussianSource


def run(n, conformal, half, use_gpu=False, periods=25):
    t0 = time.time()
    b, kp = 0.2, 4.0
    k_ref = mie.mie_a1_pole(b, kp)
    metal = DispersiveMaterial(1.0, [Drude(kp * c, 0.0)])
    h = 2 * b / n
    m = int(np.ceil(2 * half / h))
    bounds = np.repeat([[-m * h / 2, m * h / 2]], 3, axis=0).ravel()
    ball = pv.Sphere(radius=b, theta_resolution=300, phi_resolution=300)
    grid = wakis.GridFIT3D(*bounds, m, m, m, stl_solids={"ball": ball},
                           stl_materials={"ball": metal}, conformal=conformal,
                           verbose=0)  # fmt: skip
    solver = wakis.SolverFIT3D(grid, bc_low=["cpml"] * 3, bc_high=["cpml"] * 3,
                               n_pml=10, use_stl=True, use_gpu=use_gpu,
                               verbose=0)  # fmt: skip
    f0 = kp / np.sqrt(3) * c / (2 * np.pi)
    pulse = GaussianSource(frequency=f0, fwidth=0.3 * f0)
    src = CurrentSource(pulse, component="z", center=(0, 0, 1.3 * b))
    iz = (m // 2, m // 2,
          int(np.argmin(abs(solver.grid.z[:-1] + h / 2 + 1.3 * b))), "z")  # fmt: skip
    trace = []

    class Probe:
        def update(self, solver, t):
            trace.append(float(solver.E[iz]))

    Nt = int((pulse.end_time + periods / f0) / solver.dt)
    solver.emsolve(Nt, source=src, monitors=[Probe()])
    start = int(np.ceil(pulse.end_time / solver.dt))
    res = harminv(np.array(trace[start:]), solver.dt, 0.75 * f0, 1.0 * f0)
    F = 2 / solver.dt * np.sin(np.pi * res.complex_frequency * solver.dt) / (2 * np.pi)
    k = 2 * np.pi * F / c
    amp = np.abs(res.amplitude)
    i = np.argmax(amp) if amp.size else None
    kk = [np.nan, np.nan] if i is None else [k[i].real, k[i].imag]
    return dict(n=n, conformal=str(conformal), half=half, cells=m**3, Nt=Nt,
                k=kk, k_ref=[k_ref.real, k_ref.imag],
                err_re=kk[0] / k_ref.real - 1, err_im=kk[1] / k_ref.imag - 1,
                modes=[[x.real, x.imag] for x in k],
                amps=amp.tolist(), time=time.time() - t0)  # fmt: skip


if __name__ == "__main__":
    conf = False if sys.argv[2] == "False" else sys.argv[2]
    out = run(int(sys.argv[1]), conf, float(sys.argv[3]), use_gpu="--gpu" in sys.argv)
    print(json.dumps(out))
    files = [a for a in sys.argv[4:] if a != "--gpu"]
    if files:
        with open(files[0], "a") as fh:
            fh.write(json.dumps(out) + "\n")
