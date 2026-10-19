"""
Dielectric rod of `tests/test_035_tensor_averaging.py` (eps_r = 6, radius
R/2, off-centre in the square PEC box [-R, R]^2, TE n = 0 mode, E tangential
to the interface and oblique to the grid) with the permittivity realized by a
lossless Lorentz pole: eps(w) = 1 + d w0^2 / (w0^2 - w^2), w0 = 3 w_ref,
d chosen so that eps(w_ref) = 6. The dispersive problem then has the static
problem's mode as its fixed point, k R = 1.9746767234 (Delaunay FIT
reference, test_035). The leapfrog error is removed exactly (lossless Lorentz:
the scheme equals the continuous one at 2 sin(w dt / 2) / dt).

Usage: python rod_lorentz.py <n cells across 2R> <pec|all|tensor|coupled> [out.jsonl]
"""

import json
import sys
import time

import numpy as np
import pyvista as pv
from scipy.constants import c

import wakis
from wakis.dispersive import DispersiveMaterial, Lorentz
from wakis.resonances import harminv
from wakis.sources import CurrentSource, GaussianSource

R, KR = 0.1, 1.9746767234
CENTER = (0.0031, -0.0017)


def run(n, conformal, periods=200):
    t0 = time.time()
    f_ref = KR / R * c / (2 * np.pi)
    w = 2 * np.pi * f_ref
    mat = DispersiveMaterial(1.0, [Lorentz(5 * (1 - 1 / 9), 3 * w, 0.0)], name="L")
    h = 2 * R / n
    rod = pv.Cylinder(center=(*CENTER, 0), direction=(0, 0, 1), radius=R / 2,
                      height=20 * h, resolution=2880).triangulate()  # fmt: skip
    grid = wakis.GridFIT3D(
        -R, R, -R, R, -h / 2, h / 2, n, n, 1,
        stl_solids={"rod": rod}, stl_materials={"rod": mat},
        conformal="tensor" if conformal == "coupled" else conformal, verbose=0,
    )  # fmt: skip
    solver = wakis.SolverFIT3D(
        grid, bc_low=["pec", "pec", "periodic"], bc_high=["pec", "pec", "periodic"],
        use_stl=True, verbose=0,
    )  # fmt: skip
    if conformal == "coupled":
        from coupled import make_coupled

        make_coupled(solver)
        periods = 30  # the coupled variant is unstable (slowly, here)
    pulse = GaussianSource(frequency=f_ref, fwidth=0.25 * f_ref)
    src = CurrentSource(pulse, component="x", center=(0.013, 0.071, 0.0))
    idx = (int(np.argmin(abs(solver.grid.x[:-1] + 0.067))),
           int(np.argmin(abs(solver.grid.y[:-1] + h / 2 - 0.021))), 0, "y")  # fmt: skip
    values = []

    class Probe:
        def update(self, solver, t):
            values.append(float(solver.E[idx]))

    Nt = int((pulse.end_time + periods / f_ref) / solver.dt)
    solver.emsolve(Nt, source=[src], monitors=[Probe()])
    start = int(np.ceil(pulse.end_time / solver.dt))
    sig = np.array(values[start:])
    step = max(1, int(1 / (8 * f_ref * solver.dt)))
    res = harminv(sig[::step], step * solver.dt, 0.95 * f_ref, 1.05 * f_ref,
                  t0=(start + 1) * solver.dt)  # fmt: skip
    amp = np.abs(res.amplitude)
    i = np.argmax(amp)
    f = 2 / solver.dt * np.sin(np.pi * res.frequency * solver.dt) / (2 * np.pi)
    return dict(n=n, conformal=conformal, err=float(f[i] / f_ref - 1),
                decay=float(res.decay_rate[i]), modes=(f / f_ref - 1).tolist(),
                amps=(amp / amp[i]).tolist(), Nt=Nt, time=time.time() - t0)  # fmt: skip


if __name__ == "__main__":
    out = run(int(sys.argv[1]), sys.argv[2])
    print(json.dumps(out))
    if len(sys.argv) > 3:
        with open(sys.argv[3], "a") as fh:
            fh.write(json.dumps(out) + "\n")
