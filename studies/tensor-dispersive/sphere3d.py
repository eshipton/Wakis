"""
3D Drude sphere: dipole (TM_1) surface-plasmon resonance by time-domain
ring-down and harmonic inversion.

case 'cavity': lossless Drude sphere (radius b = a/2, k_p a = 6) concentric
    in a PEC sphere (radius a); TM_1 mode k a = 4.52995 (eps = -0.754),
    exact Riccati-Bessel matching (mie.cavity_tm). Conformal PEC wall.
case 'open': isolated Drude sphere (radius b = 0.2, k_p = 4, lossless) in
    vacuum with CPML; complex dipole pole of the Mie coefficient a_1
    (mie.mie_a1_pole): k b = 0.4287 + 0.0110 i.

Usage: python sphere3d.py <case> <n cells across 2b> <conformal> [out.jsonl] [--gpu]
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

CENTER = np.array([0.0071, -0.0043, 0.0029])
# Half width of the open box [m]: the CPML must be >~ lambda/2 (1.5 m) from
# the sphere, or the radiative damping (Im k) is wrong by ~2x (0.75 m: 0.029
# instead of 0.055)
HALF_OPEN = 1.5


def run(case, n, conformal, use_gpu=False, periods=None):
    t0 = time.time()
    if case == "cavity":
        a, b, kp = 1.0, 0.5, 6.0
        k_ref = mie.cavity_tm(a, b, kp, 1)
        k_ref = complex(k_ref[np.argmin(abs(k_ref - 4.53))])
        half, bc, bg = 1.1 * a, "pec", "pec"
        periods = periods or 150
    else:
        b, kp = 0.2, 4.0
        k_ref = mie.mie_a1_pole(b, kp)
        half, bc, bg = HALF_OPEN, "cpml", "vacuum"
        periods = periods or 25
    f_ref = k_ref.real * c / (2 * np.pi)
    h = 2 * b / n
    m = int(np.ceil(2 * half / h))
    lo = CENTER - m * h / 2
    hi = CENTER + m * h / 2

    def sphere(r):
        return pv.Sphere(radius=r, center=CENTER, theta_resolution=400,
                         phi_resolution=400)  # fmt: skip

    drude = DispersiveMaterial(1.0, [Drude(kp * c, 0.0)], name="drude")
    solids = {"core": sphere(b)}
    mats = {"core": drude}
    if case == "cavity":
        solids = {"cavity": sphere(a), "core": sphere(b)}
        mats = {"cavity": "vacuum", "core": drude}
    grid = wakis.GridFIT3D(
        lo[0], hi[0], lo[1], hi[1], lo[2], hi[2], m, m, m,
        stl_solids=solids, stl_materials=mats, conformal=conformal, verbose=0,
    )  # fmt: skip
    solver = wakis.SolverFIT3D(
        grid, bc_low=[bc] * 3, bc_high=[bc] * 3, bg=bg, use_stl=True,
        n_pml=10, use_gpu=use_gpu, verbose=0,
    )  # fmt: skip
    # open: a short pulse, so that the ring-down (Q ~ 20) is long after it
    pulse = GaussianSource(
        frequency=f_ref, fwidth=(0.8 if case == "open" else 0.3) * f_ref
    )
    r = 1.3 * b
    pts = [([0, 0, r], 1.0), ([0, 0, -r], 1.0), ([r, 0, 0], -0.5), ([-r, 0, 0], -0.5)]
    srcs = [CurrentSource(pulse, component="z", center=tuple(CENTER + p), amplitude=w)
            for p, w in pts]  # fmt: skip

    class Probe:
        def __init__(self, point):
            coords = [
                solver.grid.x[:-1],
                solver.grid.y[:-1],
                solver.grid.z[:-1] + h / 2,
            ]
            self.index = (
                *(int(np.argmin(abs(q - p))) for q, p in zip(coords, point)),
                "z",
            )
            self.values = []

        def update(self, solver, t):
            self.values.append(float(solver.E[self.index]))

    probes = [Probe(CENTER + p) for p, _ in pts]
    Nt = int((pulse.end_time + periods / f_ref) / solver.dt)
    solver.emsolve(Nt, source=srcs, monitors=probes)
    start = int(np.ceil(pulse.end_time / solver.dt))
    sig = sum(w * np.array(p.values[start:]) for p, (_, w) in zip(probes, pts))
    step = max(1, int(1 / (8 * f_ref * solver.dt)))
    res = harminv(sig[::step], step * solver.dt, 0.8 * f_ref, 1.2 * f_ref,
                  t0=(start + 1) * solver.dt)  # fmt: skip
    amp = np.abs(res.amplitude)
    if amp.size == 0:  # nothing resolved: keep the signal for inspection
        return dict(case=case, n=n, conformal=str(conformal), cells=m**3,
                    k=[np.nan, np.nan], k_ref=[k_ref.real, k_ref.imag], modes=[],
                    amps=[], sig=sig[::step].tolist(), sig_dt=step * solver.dt,
                    sig_t0=(start + 1) * solver.dt, time=time.time() - t0)  # fmt: skip
    i = np.argmax(amp)
    # exact removal of the leapfrog error for a lossless Drude medium
    F = res.complex_frequency
    F = 2 / solver.dt * np.sin(np.pi * F * solver.dt) / (2 * np.pi)
    k = 2 * np.pi * F / c
    return dict(case=case, n=n, conformal=str(conformal), cells=m**3, dt=solver.dt,
                Nt=Nt, k=[k[i].real, k[i].imag], k_ref=[k_ref.real, k_ref.imag],
                err_re=float(k[i].real / k_ref.real - 1),
                err_im=float(k[i].imag / k_ref.imag - 1) if k_ref.imag else None,
                modes=(k.real / k_ref.real - 1).tolist(), amps=(amp / amp[i]).tolist(),
                sig=sig[::step].tolist(), sig_dt=step * solver.dt,
                sig_t0=(start + 1) * solver.dt, time=time.time() - t0)  # fmt: skip


if __name__ == "__main__":
    case, n, conf = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    conf = False if conf == "False" else conf
    out = run(case, n, conf, use_gpu="--gpu" in sys.argv)
    print(json.dumps(out))
    files = [a for a in sys.argv[4:] if a != "--gpu"]
    if files:
        with open(files[0], "a") as fh:
            fh.write(json.dumps(out) + "\n")
