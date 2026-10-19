"""Before/after cost of the plane-wave source of UnstructuredFIT2D.

'direct': evaluate s(t - tau) on every target each step (the original code);
'table':  separable-delay table (`_build_pw_table`); 'none': a point source
only (baseline cost of the field update). Prints ns per
(unknown x step) of the full time step and the max relative difference of
the fields after the run.

    python te_source_timing.py RES [POL] [STEPS]
"""

import sys
import time

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import graded_study as G  # noqa: E402
from mie_mesh_study import LAM_MIN_DIEL, N_CYL, A, pulse  # noqa: E402

from wakis.unstructured import UnstructuredFIT2D, graded_disk_mesh  # noqa: E402

res = int(sys.argv[1])
pol = sys.argv[2] if len(sys.argv) > 2 else "TE"
steps = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
h = LAM_MIN_DIEL / res
mesh = graded_disk_mesh([A, 2.7e-6], G.size_field(h, True))


def direct(self, n):
    pw = self.plane_wave
    if self.pol == "TE":
        e = self._incident_signal(pw["pos"], n * self.dt)
        return np.bincount(pw["e"], pw["coef"] * e, minlength=self.mesh.n_edges)
    out = np.zeros(self.free.size)
    out[pw["idx"]] = pw["w"] * self._incident_signal(pw["pos"], n * self.dt) / self.dt
    return out


out = {}
for mode in ("none", "direct", "table"):
    s = UnstructuredFIT2D(mesh, {0: [N_CYL**2, 1.0]}, polarization=pol, bc="abc")
    if mode != "none":
        s.add_plane_wave(pulse())
    else:
        s.add_point_source([0.5e-6, 0.0], pulse())
    if mode == "direct":
        s._pw_values = direct.__get__(s)
    s.run(10)
    t0 = time.perf_counter()
    s.run(steps)
    dt_wall = time.perf_counter() - t0
    dofs = s.free.size + mesh.n_edges
    out[mode] = s.u.copy()
    print(
        f"{pol} res {res} {mode:6s}: {dt_wall / steps * 1e3:.2f} ms/step, "
        f"{dt_wall / (steps * dofs) * 1e9:.2f} ns per unknown-step ({dofs:,} unknowns)"
    )
d = np.abs(out["table"] - out["direct"]).max() / np.abs(out["direct"]).max()
print(f"max |table - direct| / max |direct| = {d:.1e}")
