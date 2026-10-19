"""
Memory vs recompute time of the adjoint memory ladder on a 2D waveguide bend.

The problem has the size of the ceviche waveguide-bend challenge at r = 1
(313 x 313 cells, 160 x 160 design cells = 25600 design edges) and is set
up natively in Wakis: a dielectric guide (eps 6) enters from -x, a second
one leaves to +y, the design region sits at the corner, CPML all around,
a Gaussian line current in the input guide and a DFT probe in the output
guide. The objective is |Ez(f0)|^2 at the probe.

For every rung/setting the script records: forward and backward wall time,
predicted peak bytes of the rung (`sim.memory_stats`), measured high-water
mark (tracemalloc on the host; a CuPy memory hook on the GPU), recomputed
steps and the gradient deviation from the stored-increment gradient.

Usage (cloud VM; the stored float64 rows alone are Nt x 25600 x 8 bytes)::

    python bend2d.py --n 313 --design 160 --nt 20000 --gpu --out bend_gpu.json
"""

import argparse
import json
import time
import tracemalloc

import numpy as np

import wakis
from wakis.inverse import AdjointSimulation, MaterialGrid
from wakis.sources import CurrentSource, GaussianSource

F0 = 3e9
LAM = 3e8 / F0


def make_problem(n, n_design, Nt, use_gpu, n_pml=12, cpl=40):
    dx = LAM / cpl
    L = n * dx

    def make_solver():
        grid = wakis.GridFIT3D(-L / 2, L / 2, -L / 2, L / 2, -dx / 2, dx / 2,
                               n, n, 1, verbose=0)  # fmt: skip
        solver = wakis.SolverFIT3D(
            grid, bc_low=["cpml", "cpml", "periodic"],
            bc_high=["cpml", "cpml", "periodic"], n_pml=n_pml,
            use_gpu=use_gpu, verbose=0,
        )  # fmt: skip
        w = 6 * dx
        for center, size in (((-L / 4, 0, 0), (L / 2, w, 0)),
                             ((0, L / 4, 0), (w, L / 2, 0))):  # fmt: skip
            guide = MaterialGrid(center, size, (1, 1, 1), eps_min=6.0, eps_max=6.0)
            guide.apply(solver, np.ones((1, 1, 1)))
        return solver

    D = n_design * dx
    design = MaterialGrid((0, 0, 0), (D, D, 0), (n_design // 4, n_design // 4, 1),
                          eps_min=1.0, eps_max=6.0)  # fmt: skip
    src = CurrentSource(GaussianSource(F0, 0.5 * F0), component="z",
                        center=(-L / 2 + (n_pml + 8) * dx, 0, 0))  # fmt: skip
    probe = wakis.DFTMonitor([F0], components=["Ez"],
                             center=(0, L / 2 - (n_pml + 8) * dx, 0))  # fmt: skip
    sim = AdjointSimulation(make_solver, [src], {"probe": probe}, design, Nt)
    return sim


class _GpuPeak:
    def __init__(self):
        from cupy.cuda import memory_hook

        class Hook(memory_hook.MemoryHook):
            name = "peak"

            def __init__(self):
                self.cur = self.peak = 0
                self.sizes = {}

            def malloc_postprocess(self, device_id, size, mem_size, mem_ptr, pmem_id):
                self.sizes[mem_ptr] = mem_size
                self.cur += mem_size
                self.peak = max(self.peak, self.cur)

            def free_postprocess(self, device_id, mem_size, mem_ptr, pmem_id):
                self.cur -= self.sizes.pop(mem_ptr, mem_size)

        self.hook = Hook()


def run(sim, rho, G, mode, use_gpu, **kw):
    sim.memory = mode
    sim.checkpoints = kw.get("checkpoints")
    sim.block = kw.get("block")
    sim.decimation = kw.get("decimation")
    sim.memory_budget_gb = kw.get("budget_gb")
    sim.store_dtype = kw.get("store_dtype", np.float64)
    sim.store_on_host = kw.get("on_host")
    gpu = _GpuPeak() if use_gpu else None
    if use_gpu:
        import cupy as cp

        cp.get_default_memory_pool().free_all_blocks()
        gpu.hook.__enter__()
    tracemalloc.start()
    t0 = time.perf_counter()
    sim.forward(rho)
    t1 = time.perf_counter()
    grad = sim.vjp(G)
    t2 = time.perf_counter()
    host_peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    if use_gpu:
        gpu.hook.__exit__(None, None, None)
    return grad, dict(
        mode=mode, settings={k: str(v) for k, v in kw.items()},
        t_forward=t1 - t0, t_backward=t2 - t1,
        host_peak=host_peak, gpu_peak=gpu.hook.peak if use_gpu else 0,
        **{f"rung_{k}": v for k, v in sim.memory_stats.items()},
    )


CONFIGS = [
    ("store", {}),
    ("store", dict(store_dtype=np.float32)),
    ("checkpoint", dict(checkpoints=2)),
    ("checkpoint", dict(checkpoints=4)),
    ("checkpoint", dict(checkpoints=8)),
    ("checkpoint", dict(checkpoints=16)),
    ("checkpoint", dict(checkpoints=32)),
    ("checkpoint", dict(checkpoints=16, block=1)),
    ("checkpoint", dict(checkpoints=64, block=1)),
    ("auto", dict(budget_gb=0.25)),
    ("auto", dict(budget_gb=1.0)),
    ("reverse", {}),
    ("nyquist", dict(decimation=8)),
    ("nyquist", dict(decimation=32)),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=313)
    ap.add_argument("--design", type=int, default=160)
    ap.add_argument("--nt", type=int, default=20000)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--out", default="bend.json")
    ap.add_argument("--configs", default="", help="comma-separated CONFIGS indices")
    a = ap.parse_args()
    sim = make_problem(a.n, a.design, a.nt, a.gpu)
    rho = np.random.default_rng(0).uniform(0.3, 0.7, sim.design.shape)
    out = sim.forward(rho)
    G = {"probe": {"Ez": 2e20 * out["probe"]["Ez"]}}
    results, g_ref = [], None
    pick = [int(i) for i in a.configs.split(",")] if a.configs else range(len(CONFIGS))
    for mode, kw in (CONFIGS[i] for i in pick):
        if a.gpu and mode == "store" and not kw:
            kw = dict(on_host=True)  # float64 rows on the host
        grad, rec = run(sim, rho, G, mode, a.gpu, **kw)
        if g_ref is None:
            g_ref = grad
        rec["grad_rel_err"] = float(np.max(np.abs(grad - g_ref)) / np.max(np.abs(g_ref)))
        rec["bitwise"] = bool(np.array_equal(grad, g_ref))
        print(json.dumps(rec), flush=True)
        results.append(rec)
    meta = dict(n=a.n, design=a.design, Nt=a.nt, gpu=a.gpu,
                design_edges=int(sim.design.flat_idx.size))  # fmt: skip
    with open(a.out, "w") as f:
        json.dump({"meta": meta, "results": results}, f, indent=1)


if __name__ == "__main__":
    main()
