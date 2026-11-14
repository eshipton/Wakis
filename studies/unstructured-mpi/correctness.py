"""
Distributed vs serial unstructured FIT: per-step difference against the
derived round-off bound, and the partition / halo of the test mesh.

  python studies/unstructured-mpi/correctness.py --out studies/unstructured-mpi/out/correctness.json

Runs the in-process `LocalCluster` (the same per-rank arithmetic and halo
copies as `DistributedFIT` under MPI) on:

* ``ball_mesh([0.5, 1.0], 0.22)``, Debye-Lorentz inner ball, lossy shell,
  Silver-Mueller boundary, massless edges eliminated (default tolerance):
  the distributed step uses the fused C_eff = C_f - C_s K_ss^-1 K_sf, so
  it must equal the host `DeviceStepper` bitwise and the serial LU
  elimination within `gpu.agreement_bound` (energy norm);
* the same mesh with ``massless_tol=0`` and plain dielectric: must equal
  `one_step` bitwise;
* a drop-one-halo-entry mutation of the first case (must exceed the bound).

Also records the cell partition (METIS if available, RCB, RIB) for the
figure of the partition surface.
"""

import argparse
import json
import pathlib

import numpy as np
from scipy.constants import c

HERE = pathlib.Path(__file__).resolve().parent


class Pulse:
    def __init__(self, f0, bw):
        self.s = 1 / (2 * np.pi * bw)
        self.f0 = f0

    def __call__(self, t):
        t0 = 5 * self.s
        return np.exp(-(((t - t0) / self.s) ** 2) / 2) * np.sin(2 * np.pi * self.f0 * (t - t0))


def solver(massless=True):
    from wakis.dispersive import Debye, DispersiveMaterial, Lorentz
    from wakis.unstructured import UnstructuredFIT3D, ball_mesh

    debye = DispersiveMaterial(2.0, [Debye(1.5, 0.5 / c), Lorentz(0.6, 2.0 * c, 0.3 * c)])
    mesh = ball_mesh([0.5, 1.0], 0.22)
    if massless:
        s = UnstructuredFIT3D(mesh, {0: debye, 1: [1.0, 1.0, 1e-3]}, bc="abc")
    else:
        s = UnstructuredFIT3D(mesh, {0: [3.0, 1.0]}, bc="abc", massless_tol=0.0)
    s.add_point_dipole([0.6, 0.05, 0.2], [0.2, 0.1, 1.0], Pulse(0.25 * c, 0.15 * c))
    return s


def wnorm(W, du, db):
    return float(np.sqrt(W["u"] @ du**2 + W["b"] @ db**2))


def main():
    from wakis.unstructured import gpu as G
    from wakis.unstructured import mpi as M

    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--parts", type=int, default=4)
    ap.add_argument("--out", default=str(HERE / "out" / "correctness.json"))
    a = ap.parse_args()
    n, P = a.steps, a.parts
    out = {"steps": n, "parts": P}

    s = solver(True)
    bound = G.agreement_bound(solver(True), n, probes=False)
    W = bound["weights"]
    cl = M.LocalCluster(s, P, partitioner="rcb")
    bad = M.LocalCluster(solver(True), P, partitioner="rcb", _mutation="drop_halo")
    dev = G.DeviceStepper(solver(True), xp=np)
    d_lu, d_dev, d_bad, norm = [], [], [], []
    for _ in range(n):
        cl.step()
        bad.step()
        dev.step()
        s.one_step()
        g, gb, st = cl.gather_state(), bad.gather_state(), dev.get_state()
        d_lu.append(wnorm(W, g["u"] - s.u, g["f"] - s.b))
        d_dev.append(wnorm(W, g["u"] - st["u"], g["f"] - st["b"]))
        d_bad.append(wnorm(W, gb["u"] - s.u, gb["f"] - s.b))
        norm.append(wnorm(W, s.u, s.b))
    out["massless"] = dict(
        n_massless=int(s.massless.size), n_cells=s.mesh.n_cells, eps=bound["eps"].tolist(),
        K=bound["K"], depth=bound["depth"], d_serial_lu=d_lu, d_fused_host=d_dev,
        d_mutation=d_bad, norm=norm, comm=cl.comm_stats(),
    )  # fmt: skip
    print("massless: max d/eps", max(np.array(d_lu) / bound["eps"]), "fused max", max(d_dev),
          "mutation final/eps", d_bad[-1] / bound["eps"][-1])  # fmt: skip

    s0 = solver(False)
    cl0 = M.LocalCluster(s0, P, partitioner="rcb")
    d0 = []
    for _ in range(n):
        cl0.step()
        s0.one_step()
        g = cl0.gather_state()
        d0.append(float(max(np.abs(g["u"] - s0.u).max(), np.abs(g["f"] - s0.b).max())))
    out["plain"] = dict(max_abs_diff=d0, bitwise=bool(max(d0) == 0.0))
    print("plain: bitwise", max(d0) == 0.0)

    mesh = s.mesh
    cent = mesh.nodes[mesh.cells].mean(1)
    parts = {}
    for m in ("metis", "rcb", "rib"):
        try:
            p = M.partition_cells(mesh, 8, m)
        except ImportError:
            continue
        parts[m] = dict(parts=p.tolist(), stats=M.partition_stats(mesh, p))
        print(m, parts[m]["stats"]["cut_facets"])
    out["partition"] = dict(centroids=cent.tolist(), parts=parts)
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(a.out).write_text(json.dumps(out))


if __name__ == "__main__":
    main()
