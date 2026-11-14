"""
Strong and weak scaling of the MPI-distributed unstructured FIT step
(`wakis.unstructured.mpi.DistributedFIT`) on one many-core machine.

  python studies/unstructured-mpi/scaling.py --suite all --out studies/unstructured-mpi/out/scaling.json
  mpiexec -n 4 python studies/unstructured-mpi/scaling.py --one '{"P": 4, "h": 0.05}'

Every configuration runs in its own ``mpiexec -n P`` process group (rank 0
prints one JSON line). Per configuration:

* mesh: `ball_mesh([0.5, 1.0], h)` (unit ball, eps_r = 3 inner ball, PEC
  outer sphere), Voronoi Hodge with massless-edge elimination (default
  ``massless_tol``), one point dipole. Built on rank 0 only; the pieces are
  scattered (``mode="scatter"``).
* ``P = 0`` is the **baseline**: the same solver's serial `one_step`
  (SciPy CSR SpMV, one process, OMP/MKL/OPENBLAS threads = 1) on the same
  mesh. ``P >= 1`` is `DistributedFIT` with P ranks (P = 1: no halo, the
  distributed code path's own overhead).
* timing: 5 warm-up steps, barrier, then steps until >= `--seconds` (at
  least 20 steps); the wall time per step is the max over ranks. Each rank
  also accumulates its compute and exchange (Isend/Irecv/Waitall) time.
* communication: halo doubles received per step on each rank (sum over the
  u and b vectors), bytes sent per step, partition surface (cut facets) and
  load imbalance (max/mean owned dofs).
* preflight: `estimate_memory(n_cells)` on rank 0 before the build.
"""

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

import numpy as np

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, "1")

HERE = pathlib.Path(__file__).resolve().parent


class Pulse:
    """Picklable Gaussian-modulated sine."""

    def __init__(self, f0):
        self.s = 1 / (2 * np.pi * 0.5 * f0)
        self.f0 = f0

    def __call__(self, t):
        t0 = 5 * self.s
        return np.exp(-(((t - t0) / self.s) ** 2) / 2) * np.sin(2 * np.pi * self.f0 * (t - t0))


def build(h):
    from scipy.constants import c

    from wakis.unstructured import UnstructuredFIT3D, ball_mesh

    mesh = ball_mesh([0.5, 1.0], h)
    s = UnstructuredFIT3D(mesh, {0: [3.0, 1.0]}, bc="pec")
    s.add_point_dipole([0.1, 0.05, 0.2], [0.2, 0.1, 1.0], Pulse(0.3 * c))
    return s


def _time_steps(step, seconds, sync=lambda: None):
    for _ in range(5):
        step()
    sync()
    t0 = time.perf_counter()
    n = 0
    while True:
        step()
        n += 1
        if n >= 20 and time.perf_counter() - t0 >= seconds:
            break
    sync()
    return n, time.perf_counter() - t0


def one(cfg):
    P, h = int(cfg["P"]), float(cfg["h"])
    seconds = float(cfg.get("seconds", 4.0))
    method = cfg.get("method", "auto")
    if P == 0:  # serial SciPy baseline
        from wakis.unstructured import estimate_memory

        t0 = time.perf_counter()
        s = build(h)
        t_setup = time.perf_counter() - t0
        n, wall = _time_steps(s.one_step, seconds)
        return dict(cfg, n_cells=s.mesh.n_cells, n_unknowns=int(s.free.size + s.mesh.n_faces),
                    n_massless=int(s.massless.size), t_setup=t_setup, steps=n,
                    t_step=wall / n, est_bytes=estimate_memory(s.mesh))  # fmt: skip

    from mpi4py import MPI

    from wakis.unstructured import estimate_memory
    from wakis.unstructured.mpi import DistributedFIT, partition_cells, partition_stats

    comm = MPI.COMM_WORLD
    assert comm.Get_size() == P, (comm.Get_size(), P)
    me = comm.Get_rank()
    s, info = None, {}
    t0 = time.perf_counter()
    if me == 0:
        s = build(h)
        info["t_build"] = time.perf_counter() - t0
        info["n_cells"] = s.mesh.n_cells
        info["n_unknowns"] = int(s.free.size + s.mesh.n_faces)
        info["n_massless"] = int(s.massless.size)
        info["est_bytes"] = estimate_memory(s.mesh)
        t1 = time.perf_counter()
        parts = partition_cells(s.mesh, P, method)
        info["t_partition"] = time.perf_counter() - t1
        info["partition"] = partition_stats(s.mesh, parts)
    else:
        parts = None
    parts = comm.bcast(parts, root=0)
    t1 = time.perf_counter()
    d = DistributedFIT(s, comm, parts=parts, mode="scatter", timers=True)
    comm.Barrier()
    info["t_distribute"] = time.perf_counter() - t1
    del s
    n, wall = _time_steps(d.step, seconds, comm.Barrier)
    # timers include the warm-up: rescale to the timed steps
    d.t = {"compute": 0.0, "exchange": 0.0}
    d.bytes_sent = 0
    n2, wall2 = _time_steps(d.step, seconds, comm.Barrier)
    tot = n2 + 5
    loc = dict(
        compute=d.t["compute"] / tot,
        exchange=d.t["exchange"] / tot,
        bytes=d.bytes_sent / tot,
        halo={k: sp_.halo_size for k, sp_ in d.r.spaces.items()},
        own={k: sp_.n_own for k, sp_ in d.r.spaces.items()},
        neighbours=len(set().union(*[set(sp_.neighbours) for sp_ in d.r.spaces.values()])),
    )
    ranks = comm.gather(loc, root=0)
    wall_max = comm.reduce(wall2 / n2, op=MPI.MAX, root=0)
    if me != 0:
        return None
    own_tot = [sum(r["own"].values()) for r in ranks]
    return dict(
        cfg, **info, steps=n2, t_step=wall_max,
        t_compute_max=max(r["compute"] for r in ranks),
        t_compute_mean=float(np.mean([r["compute"] for r in ranks])),
        t_exchange_max=max(r["exchange"] for r in ranks),
        t_exchange_mean=float(np.mean([r["exchange"] for r in ranks])),
        halo_total=int(sum(sum(r["halo"].values()) for r in ranks)),
        halo_max=int(max(sum(r["halo"].values()) for r in ranks)),
        bytes_total=float(sum(r["bytes"] for r in ranks)),
        neighbours_max=int(max(r["neighbours"] for r in ranks)),
        dof_imbalance=float(max(own_tot) / np.mean(own_tot)),
        ranks=ranks,
    )  # fmt: skip


def suite(name, h_strong, h_weak1, ps):
    cfgs = []
    if name in ("strong", "all"):
        cfgs += [dict(study="strong", P=0, h=h_strong)]
        cfgs += [dict(study="strong", P=p, h=h_strong) for p in ps]
    if name in ("weak", "all"):
        for p in ps:
            hp = h_weak1 / p ** (1 / 3)
            cfgs += [dict(study="weak", P=0, h=hp), dict(study="weak", P=p, h=hp)]
    if name in ("partitioners", "all"):
        for p in (8, 16):
            for m in ("metis", "rcb", "rib"):
                cfgs += [dict(study="partitioners", P=p, h=h_strong, method=m)]
    return cfgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--one")
    ap.add_argument("--suite", default="all")
    ap.add_argument("--h-strong", type=float, default=0.04)
    ap.add_argument("--h-weak1", type=float, default=0.09)
    ap.add_argument("--ps", default="1,2,4,8,16,32")
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--out", default=str(HERE / "out" / "scaling.json"))
    a = ap.parse_args()
    if a.one:
        res = one(json.loads(a.one))
        if res is not None:
            print("RESULT " + json.dumps(res), flush=True)
        return
    ps = [int(x) for x in a.ps.split(",")]
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(out.read_text()) if out.exists() else []
    done = {json.dumps({k: r[k] for k in ("study", "P", "h", "method") if k in r}, sort_keys=True)
            for r in results}  # fmt: skip
    for cfg in suite(a.suite, a.h_strong, a.h_weak1, ps):
        cfg["seconds"] = a.seconds
        key = json.dumps({k: cfg[k] for k in ("study", "P", "h", "method") if k in cfg},
                         sort_keys=True)  # fmt: skip
        if key in done:
            continue
        P = cfg["P"]
        cmd = [sys.executable, __file__, "--one", json.dumps(cfg)]
        if P >= 1:
            cmd = ["mpiexec", "--oversubscribe", "--bind-to", "core" if P <= 16 else "hwthread",
                   "-n", str(P)] + cmd  # fmt: skip
        t0 = time.perf_counter()
        p = subprocess.run(cmd, capture_output=True, text=True)
        line = [x for x in p.stdout.splitlines() if x.startswith("RESULT ")]
        if p.returncode or not line:
            print("FAILED", cfg, p.returncode, p.stdout[-2000:], p.stderr[-4000:], flush=True)
            continue
        res = json.loads(line[0][7:])
        res["t_wall_run"] = time.perf_counter() - t0
        results.append(res)
        out.write_text(json.dumps(results, indent=1))
        print(f"{cfg['study']:>12} P={P:>2} h={cfg['h']:.4f} {cfg.get('method', '')}"
              f" cells={res['n_cells']} t_step={res['t_step'] * 1e3:.2f} ms", flush=True)  # fmt: skip


if __name__ == "__main__":
    main()
