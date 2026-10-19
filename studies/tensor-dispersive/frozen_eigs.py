"""Frozen-eps comparison: eigenvalue nearest k_ref^2 of C^T C N(k_ref)."""

import sys

import numpy as np
from frozen_proto2d import Grid, build_N
from scipy.sparse.linalg import eigs

kp = 4.0
import os

kref = float(os.environ.get("KREF", 3.7492892652627203))
ns = [int(a) for a in sys.argv[1].split(",")]
schemes = sys.argv[2].split(",")
for n in ns:
    g = Grid(n)
    A0 = (g.C.T @ g.C).tocsr()
    row = []
    for s in schemes:
        A = (A0 @ build_N(g, kref, kp, s)).tocsc()
        w = eigs(
            A, k=8, sigma=kref**2, which="LM", tol=1e-12, return_eigenvectors=False
        )
        k = np.sqrt(np.abs(w.real))
        k = np.sort(k[np.argsort(abs(k - kref))[:3]])
        e = k / kref - 1
        row.append(f"{s}:" + ",".join(f"{x:+.2e}" for x in e))
    print(n, "  ".join(row), flush=True)
