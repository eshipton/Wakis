"""
Throughput of the unstructured FIT step on the GPU (`wakis.unstructured.gpu`).

  python studies/unstructured-gpu/bench.py --dim 3 --h 0.25 0.12 ... --out out/bench.json
  python studies/unstructured-gpu/bench.py --one '{"dim": 3, "h": 0.1}'     (one size; JSON on stdout)

Every size runs in its own subprocess (clean peak RSS and device pool). Per size:

* mesh: 3D `ball_mesh([0.5, 1.0], h)`, eps_r = 3 inner ball, PEC outer sphere,
  Voronoi Hodge with massless-edge elimination (default massless_tol);
  2D `disk_mesh([0.5, 1.0], h)`, TM, eps_r = 3 inner disk, PEC.
  One point source (dipole). Unknowns = free edges + faces (3D) or free
  nodes + edges (2D): the state the step updates.
* ``cpu``: `solver.one_step` (SciPy CSR, one thread: OMP/MKL/OPENBLAS = 1),
  wall time per step after 5 warm-up steps, >= 3 s or 300 steps.
* ``gpu``: `DeviceStepper(solver).step`, after 20 warm-up steps, steps
  enqueued without synchronisation, one sync at the end (>= 1 s).
* ``gpu_mon`` / ``cpu_mon``: the same with one probe and a 3-frequency DFT
  of all node / edge dofs (monitor cost).
* ``breakdown``: per-kernel device time (CUDA events around each operation
  of the 3D / TM step, 200 steps).
* ``agree``: relative energy-norm difference GPU vs CPU after 50 equal steps.
* ``mem``: host peak RSS (ru_maxrss), device pool bytes after the warm-up,
  `DeviceStepper.device_bytes`, and `estimate_memory`.
* ``cart``: `SolverFIT3D(use_gpu=True)` on a PEC cube with 6 N^3 ~ the same
  number of unknowns, ``fused=False`` (CuPy kernels) and ``fused=True``
  (`wakis.gpu_fused.FusedStep`), no sources, 20 warm-up steps (3D only).
"""

import argparse
import json
import os
import pathlib
import resource
import subprocess
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent


def _sync():
    import cupy as cp

    cp.cuda.Device().synchronize()


def _pulse(f0):
    s = 1 / (2 * np.pi * 0.5 * f0)
    return lambda t: np.exp(-(((t - 5 * s) / s) ** 2) / 2) * np.sin(2 * np.pi * f0 * (t - 5 * s))


def build(dim, h):
    from scipy.constants import c

    from wakis.unstructured import UnstructuredFIT2D, UnstructuredFIT3D, ball_mesh, disk_mesh

    t0 = time.perf_counter()
    if dim == 3:
        mesh = ball_mesh([0.5, 1.0], h)
        t_mesh = time.perf_counter() - t0
        s = UnstructuredFIT3D(mesh, {0: [3.0, 1.0]}, bc="pec")
        s.add_point_dipole([0.1, 0.05, 0.2], [0.2, 0.1, 1.0], _pulse(0.3 * c))
        n_unk = s.free.size + mesh.n_faces
    else:
        mesh = disk_mesh([0.5, 1.0], h)
        t_mesh = time.perf_counter() - t0
        s = UnstructuredFIT2D(mesh, {0: [3.0, 1.0]}, polarization="TM", bc="pec")
        s.add_point_source([0.1, 0.05], _pulse(0.3 * c))
        n_unk = s.free.size + mesh.n_edges
    t_setup = time.perf_counter() - t0 - t_mesh
    return s, mesh, n_unk, t_mesh, t_setup


def add_monitors(s):
    from scipy.constants import c

    if s.mesh.dim == 3:
        s.add_probe([[0.3, 0.1, 0.0]], 2)
    else:
        s.add_probe([[0.3, 0.1]])
    s.add_dft([0.2 * c, 0.3 * c, 0.4 * c])


def time_cpu(s, t_min=3.0, n_max=300):
    for _ in range(5):
        s.one_step()
    n, t0 = 0, time.perf_counter()
    while n < n_max and (time.perf_counter() - t0 < t_min or n < 3):
        s.one_step()
        n += 1
    return (time.perf_counter() - t0) / n, n


def time_gpu(dev, t_min=1.0, n_min=50):
    dev.run(20)
    _sync()
    n = n_min
    while True:
        dev.prepare(n)
        for pr in dev.probes:
            dev._reserve(pr, pr["k"] + n)
        t0 = time.perf_counter()
        for _ in range(n):
            dev.step()
        _sync()
        dt = time.perf_counter() - t0
        if dt >= t_min or n >= 20000:
            return dt / n, n
        n = int(n * max(2.0, 1.2 * t_min / max(dt, 1e-6)))


def breakdown(dev, n=200):
    """Device time per operation (CUDA events), 3D or 2D TM step without groups."""
    import cupy as cp

    S, dt, m = dev.S, dev.dt, dev.m
    names = ["curl_e (SpMV)", "b/v update", "Hodge (M_nu b)", "curl_h (SpMV)", "sources",
             "capacitive update", "monitors"]  # fmt: skip
    acc = np.zeros(len(names))
    ev = [cp.cuda.Event() for _ in range(len(names) + 1)]
    for _ in range(n):
        t_half = (dev.n + 0.5) * dt
        u = S["u"]
        ev[0].record()
        if dev.kind == "3d":
            y = dev.Ceff @ u
            ev[1].record()
            dev._axpy(S["b"], m * dt, y)
            ev[2].record()
            h = dev.Mnu * S["b"]
            ev[3].record()
            r = dev.CfT @ h
        else:
            y = dev._grad(u)
            ev[1].record()
            dev._axpy(S["v"], m * dt, y)
            ev[2].record()
            h = dev.stiff * S["v"]
            ev[3].record()
            r = dev.GT @ h
        ev[4].record()
        dev._add_sources(r, "node", t_half, m)
        ev[5].record()
        dev._cap_step(r)
        ev[6].record()
        dev.n += 1
        dev._record(dev.n * dt)
        ev[7].record()
        ev[7].synchronize()
        acc += [cp.cuda.get_elapsed_time(ev[i], ev[i + 1]) for i in range(len(names))]
    return dict(zip(names, (acc / n * 1e3).tolist()))  # us per step


def agree(s, n=50):
    from wakis.unstructured.gpu import DeviceStepper

    dev = DeviceStepper(s)
    W = dev.weights()
    dev.run(n)
    for _ in range(n):
        s.one_step()
    st = dev.get_state()
    key = "b" if s.mesh.dim == 3 else "v"
    host = {"u": s.u, key: getattr(s, key)}
    d = np.sqrt(sum(np.sum(W[k] * (st[k] - host[k]) ** 2) for k in ("u", key)))
    x = np.sqrt(sum(np.sum(W[k] * host[k] ** 2) for k in ("u", key)))
    return float(d / x)


def cartesian(n_unk, fused):
    import wakis

    N = max(8, int(round((n_unk / 6) ** (1 / 3))))
    grid = wakis.GridFIT3D(-0.1, 0.1, -0.1, 0.1, -0.1, 0.1, N, N, N, verbose=0)
    s = wakis.SolverFIT3D(grid, use_gpu=True, fused=fused, verbose=0,
                          bc_low=["pec"] * 3, bc_high=["pec"] * 3)  # fmt: skip
    for _ in range(20):
        s.one_step()
    _sync()
    n, t_min = 50, 1.0
    while True:
        t0 = time.perf_counter()
        for _ in range(n):
            s.one_step()
        _sync()
        dt = time.perf_counter() - t0
        if dt >= t_min or n >= 20000:
            break
        n = int(n * max(2.0, 1.2 * t_min / max(dt, 1e-6)))
    return dict(N=N, n_unknowns=6 * s.N, s_per_step=dt / n, n_steps=n, fused=bool(fused))


def one(cfg):
    import cupy as cp

    from wakis.unstructured.estimate import estimate_memory
    from wakis.unstructured.gpu import DeviceStepper

    dim, h = int(cfg["dim"]), float(cfg["h"])
    s, mesh, n_unk, t_mesh, t_setup = build(dim, h)
    out = dict(dim=dim, h=h, n_cells=int(mesh.n_cells), n_unknowns=int(n_unk),
               n_massless=int(getattr(s, "massless", np.zeros(0)).size), t_mesh_s=t_mesh,
               t_setup_s=t_setup, est_host_bytes=float(estimate_memory(mesh)))  # fmt: skip
    pool = cp.get_default_memory_pool()
    # GPU, no monitors
    p0 = pool.used_bytes()
    t1 = time.perf_counter()
    dev = DeviceStepper(s)
    out["t_device_setup_s"] = time.perf_counter() - t1
    out["gpu_s_per_step"], out["gpu_steps"] = time_gpu(dev)
    out["dev_pool_bytes"] = int(pool.used_bytes() - p0)
    out["dev_pool_total_bytes"] = int(pool.total_bytes())
    out["dev_stepper_bytes"] = int(dev.device_bytes())
    out["breakdown_us"] = breakdown(dev)
    del dev
    pool.free_all_blocks()
    # CPU, no monitors
    if not cfg.get("skip_cpu"):
        out["cpu_s_per_step"], out["cpu_steps"] = time_cpu(s)
    out["host_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    # agreement after 50 equal steps
    out["agree_rel"] = agree(s)
    # with monitors
    add_monitors(s)
    dev = DeviceStepper(s)
    out["gpu_mon_s_per_step"], _ = time_gpu(dev)
    out["breakdown_mon_us"] = breakdown(dev)
    del dev
    pool.free_all_blocks()
    if not cfg.get("skip_cpu"):
        out["cpu_mon_s_per_step"], _ = time_cpu(s, t_min=2.0, n_max=100)
    for k in ("s_per_step", "mon_s_per_step"):
        for dev_ in ("gpu", "cpu"):
            if f"{dev_}_{k}" in out:
                out[f"{dev_}_{k.replace('s_per_step', 'ns_per_unknown_step')}"] = (
                    out[f"{dev_}_{k}"] / n_unk * 1e9
                )
    if dim == 3 and not cfg.get("skip_cart"):
        del s
        pool.free_all_blocks()
        out["cart"] = [cartesian(n_unk, fused) for fused in (False, True)]
        for c_ in out["cart"]:
            c_["ns_per_unknown_step"] = c_["s_per_step"] / c_["n_unknowns"] * 1e9
    out["gpu_name"] = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--one")
    ap.add_argument("--dim", type=int, default=3)
    ap.add_argument("--h", type=float, nargs="+")
    ap.add_argument("--skip-cpu-below-h", type=float, default=0.0,
                    help="skip the CPU timing for h below this (time budget)")
    ap.add_argument("--out", default=str(HERE / "out" / "bench.json"))
    a = ap.parse_args()
    if a.one:
        print("RESULT " + json.dumps(one(json.loads(a.one))), flush=True)
        return
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    res = []
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        res = json.loads(out.read_text())
    for h in a.h:
        cfg = dict(dim=a.dim, h=h, skip_cpu=h < a.skip_cpu_below_h)
        print(f"--- {cfg}", flush=True)
        p = subprocess.run([sys.executable, __file__, "--one", json.dumps(cfg)], env=env,
                           capture_output=True, text=True)  # fmt: skip
        sys.stdout.write(p.stdout[-4000:])
        sys.stderr.write(p.stderr[-4000:])
        line = [x for x in p.stdout.splitlines() if x.startswith("RESULT ")]
        if not line:
            print(f"size h={h} failed (rc {p.returncode})", flush=True)
            continue
        r = json.loads(line[-1][7:])
        print(json.dumps({k: v for k, v in r.items() if "ns_per" in k or k == "n_cells"}), flush=True)
        res.append(r)
        out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
