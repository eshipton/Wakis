"""
Control runs for the accuracy gain of LTS on the Mie meshes (res 10).

* lts-same-dt: LTS machinery with the coarse step equal to the global step
  (must reproduce the global error exactly if LTS adds no error);
* global-half: global stepping at half the step (cfl 0.475).

If the error falls as dt grows, the gain is the partial cancellation of
spatial (lagging) and leapfrog temporal (leading) dispersion, not LTS.
Usage: python dt_control.py {global-half|lts-same-dt} {TM|TE}
"""

import sys

import numpy as np

sys.path.insert(0, ".")
import lts_study as LS  # noqa: E402

S = LS.S
S.T_RUN = 1000e-15
from wakis.unstructured import UnstructuredFIT2D  # noqa: E402

variant = sys.argv[1]
pol = sys.argv[2]
orig = UnstructuredFIT2D.__init__


def patched(self, *a, **k):
    if variant == "global-half":
        k["cfl"] = 0.475
        k["lts"] = None
    if variant == "lts-same-dt":
        k["cfl"] = 0.95
    if variant == "lts-same-dt" and k.get("lts"):
        # coarse dt = global dt
        g = UnstructuredFIT2D.__new__(UnstructuredFIT2D)
        orig(g, *a, **{**k, "lts": None})
        k["lts"] = dict(p=2, dt=g.dt)
    orig(self, *a, **k)


UnstructuredFIT2D.__init__ = patched
mode = "global" if variant == "global-half" else "lts"
out = LS.run("tri-graded", pol, 10, mode)
q = np.array(out["q"]) / np.array(out["q_ref"]) - 1
print(
    variant,
    pol,
    "dt",
    out["dt"],
    "steps",
    out["steps"],
    "rms %.4f" % np.sqrt(np.mean(q**2)),
    flush=True,
)
