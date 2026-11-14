"""
Throughput of the tetrahedral FIT step with and without the spherical CFS-PML,
host (SciPy, one thread) vs device (CuPy `DeviceStepper`).

For each ball radius RB (edge h = 1, 10 PML layers of dr = h):

* ``ball_abc``     ball only, first-order Silver-Mueller boundary (the cheap
                    absorbing baseline: same interior, no layer),
* ``shell_abc``    ball + 10 vacuum layers, Silver-Mueller on the outside
                    (same cells as the PML case: the cost of the cells alone),
* ``shell_pml``    ball + 10-layer CFS-PML (kappa_max = 2, alpha > 0).

Reported: steps/s, cell-steps/s and the PML overhead (shell_pml / shell_abc
time per step). Device timings use CUDA events around `run(n)` after a
warm-up; host timings `solver.run(n)` with OMP/MKL threads = 1.

Usage: python bench_pml3d_gpu.py [radii ...]   -> bench.json
"""

import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402
from scipy.constants import c  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
NL = 10


def pulse(t, lam=16.0):
    tau = lam / (np.pi * c)
    x = (t - 4 * tau) / tau
    return -x * np.exp(-(x**2))


def build(kind, rb):
    from wakis.unstructured import UnstructuredFIT3D, ball_mesh
    from wakis.unstructured.pml3d import ball_with_pml_mesh

    if kind == "ball_abc":
        s = UnstructuredFIT3D(ball_mesh([rb], 1.0), bc="abc")
    else:
        m, cfg = ball_with_pml_mesh(rb, 1.0, NL)
        if kind == "shell_abc":
            s = UnstructuredFIT3D(m, bc="abc")
        else:
            s = UnstructuredFIT3D(m, bc="pec", pml=dict(cfg, kappa_max=2.0, alpha=2 * np.pi * c / 16 / 10))
    s.add_point_dipole([0.31, 0.17, 0.23], [0, 0, 1], pulse)
    s.add_probe([[rb - 1.0, 0.0, 0.0]], 2)
    return s


def time_host(s, n):
    t = time.perf_counter()
    s.run(n)
    return (time.perf_counter() - t) / n


def time_device(s, n):
    import cupy as cp

    from wakis.unstructured.gpu import DeviceStepper

    dev = DeviceStepper(s, xp=cp)
    dev.run(20)
    cp.cuda.Stream.null.synchronize()
    a, b = cp.cuda.Event(), cp.cuda.Event()
    a.record()
    dev.run(n)
    b.record()
    b.synchronize()
    return cp.cuda.get_elapsed_time(a, b) / 1e3 / n, dev.device_bytes()


def main(radii):
    import cupy as cp

    out = dict(gpu=cp.cuda.runtime.getDeviceProperties(0)["name"].decode(), layers=NL, runs=[])
    for rb in radii:
        for kind in ("ball_abc", "shell_abc", "shell_pml"):
            s = build(kind, rb)
            ext = s._pml.layout.n if s._pml is not None else s.free.size + s.mesh.n_faces
            th = time_host(s, 50)
            s2 = build(kind, rb)
            n_dev = 2000
            td, nbytes = time_device(s2, n_dev)
            r = dict(rb=rb, kind=kind, n_cells=int(s.mesh.n_cells), n_free=int(s.free.size), n_state=int(ext),
                     host_s_per_step=th, device_s_per_step=td, device_bytes=int(nbytes),
                     host_cell_steps_per_s=s.mesh.n_cells / th, device_cell_steps_per_s=s.mesh.n_cells / td,
                     speedup=th / td)  # fmt: skip
            out["runs"].append(r)
            print(json.dumps(r), flush=True)
    for rb in radii:
        R = {r["kind"]: r for r in out["runs"] if r["rb"] == rb}
        for dev in ("host", "device"):
            R["shell_pml"][f"{dev}_pml_overhead"] = (
                R["shell_pml"][f"{dev}_s_per_step"] / R["shell_abc"][f"{dev}_s_per_step"]
            )
        print(rb, "PML overhead host", R["shell_pml"]["host_pml_overhead"], "device",
              R["shell_pml"]["device_pml_overhead"])  # fmt: skip
    with open(os.path.join(HERE, "bench.json"), "w") as f:
        json.dump(out, f, indent=1)


def plot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    S = json.load(open(os.path.join(HERE, "bench.json")))
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    style = {"ball_abc": ("s", "ball, Silver-Müller"), "shell_abc": ("^", f"ball + {S['layers']} vacuum layers, Silver-Müller"),
             "shell_pml": ("o", f"ball + {S['layers']}-layer CFS-PML")}  # fmt: skip
    from matplotlib.ticker import NullFormatter

    for i, (kind, (mk, lab)) in enumerate(style.items()):
        R = [r for r in S["runs"] if r["kind"] == kind]
        n = [r["n_cells"] for r in R]
        col = f"C{i}"
        axs[0].loglog(n, [r["device_cell_steps_per_s"] for r in R], mk + "-", color=col, label=f"{lab} ({S['gpu']})")
        axs[0].loglog(n, [r["host_cell_steps_per_s"] for r in R], mk + "--", color=col, mfc="none",
                      label=f"{lab} (host SciPy, 1 thread)")  # fmt: skip
        axs[1].semilogx(n, [r["speedup"] for r in R], mk + "-", color=col, label=lab)
    for ax in axs:
        ax.xaxis.set_minor_formatter(NullFormatter())
    axs[0].set_xlabel("tetrahedra")
    axs[0].set_ylabel("cell-steps per second")
    axs[0].legend(fontsize=6)
    axs[1].set_xlabel("tetrahedra")
    axs[1].set_ylabel("device / host speed-up (same step)")
    axs[1].legend(fontsize=7)
    for ax in axs:
        ax.grid(alpha=0.3, which="both")
    fig.suptitle("Tetrahedral FIT step with the spherical CFS-PML on the GPU")
    fig.savefig(os.path.join(HERE, "..", "..", "docs", "img", "unstructured_pml3d_gpu_throughput.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "plot":
        plot()
    else:
        main([float(r) for r in sys.argv[1:]] or [8.0, 12.0, 16.0, 20.0])
