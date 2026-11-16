# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
GPU benchmark for 3D Cartesian subgridding (`wakis/subgrid3d/gpu_fused.py`).

Benchmarks the arms of `docs/physicsguide_sections/subgrid3d.md` section 9e
and `docs/physicsguide_sections/subgrid3d_gpu.md`:

* Uniform PEC box `1.0 x 0.8 x 0.6 m` loaded with an `eps_r = 9` dielectric
  cube (`studies/subgrid3d/run_study_lts.py` part D, `H = 0.1 ... 0.0125 m`):
  - `SolverFIT3D(use_gpu=True, fused=False)` (`fit3d_csr`, per-step Python)
  - `SolverFIT3D(use_gpu=True, fused=True)` (`fit3d_fused`, per-step Python `one_step()`)
  - `SolverFIT3D(use_gpu=True, fused=True)` with `FusedLoop(..., chunk=64)` (`fit3d_fused_graph`, CUDA graph)
  - `SubgridLTSFIT3D(kernels="csr", backend="cupy")` (`subgrid_csr`)
  - `SubgridLTSFIT3D(kernels="block", backend="cupy")` (`subgrid_block`)
  - `FusedSubgridStepper(solver, graph_chunk=0)` (`subgrid_fused_direct`)
  - `FusedSubgridStepper(solver, graph_chunk=64)` (`subgrid_fused_graph`)
* Loaded-cavity 2:1 and 3:1 subgrid meshes from `run_study_lts.py` part D,
  both global `dt` (`lts=False`) and local time stepping (`lts=True`):
  - `subgrid_csr`, `subgrid_block`, `subgrid_fused_direct`, `subgrid_fused_graph`.

Each arm is timed over 200 steps after warm-up using `cp.cuda.Event`. Records
the GPU model, measured device-to-device copy bandwidth, bytes moved per step,
roofline ms/step, measured ms/step, ns per unknown-step, and wall time to the
same physical duration `T = 10 ns`.

Usage:
    python studies/subgrid3d/gpu/bench_fused.py [--quick] --out DIR
"""

import argparse
import json
import os
import platform
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
STUDY_DIR = os.path.dirname(HERE)
ROOT_DIR = os.path.dirname(os.path.dirname(STUDY_DIR))
for _p in (ROOT_DIR, STUDY_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cupy as cp
from run_study_lts import BOX, make_solver, solver_axes

from wakis import GridFIT3D, SolverFIT3D
from wakis.gpu_fused import FusedLoop
from wakis.subgrid3d.gpu_fused import FusedSubgridStepper

T_PHYS = 10.0e-9


def measure_copy_bandwidth(n_elem=20_000_000, n_warmup=10, n_iter=50):
    """Measure device-to-device float64 copy bandwidth [GB/s] using `cp.cuda.Event`."""
    src = cp.ones(n_elem, dtype=np.float64)
    dst = cp.empty_like(src)
    for _ in range(n_warmup):
        cp.copyto(dst, src)
    cp.cuda.Device().synchronize()
    ev0 = cp.cuda.Event()
    ev1 = cp.cuda.Event()
    ev0.record()
    for _ in range(n_iter):
        cp.copyto(dst, src)
    ev1.record()
    ev1.synchronize()
    ms = float(cp.cuda.get_elapsed_time(ev0, ev1)) / n_iter
    bytes_copied = 2 * n_elem * 8  # read src + write dst
    gb_s = (bytes_copied * 1e-9) / (ms * 1e-3)
    del src, dst
    cp.get_default_memory_pool().free_all_blocks()
    return float(gb_s), int(bytes_copied), float(ms)


def bytes_per_step_model(stepper):
    """
    Compulsory DRAM byte traffic per coarse step [B] of `FusedSubgridStepper`:
    Faraday (`sg_faraday`) + Ampere (`sg_ampere`) + LTS Chebyshev substeps
    (`sg_lts_face`, `sg_lts_edge`).
    """
    nF = int(stepper.nF)
    nE = int(stepper.tables.nE)
    n_free = int(stepper.n_free)
    n_hang = int(stepper.tables.hang_edges.size)
    nnz_pht = int(stepper.tables.pht_val.size)

    # Faraday: read/write b (16 nF), edge-source table (24 nE), free edges e (8 n_free)
    b_faraday = 16 * nF + 24 * nE + 8 * n_free
    if stepper.has_lts:
        n_f1 = int(stepper.lts_levels[1]["nF"])
        b_faraday += 4 * n_free + 8 * n_f1
    if stepper.has_cpml:
        b_faraday += 8 * nF + 32 * int(stepper._n_psi)

    # Ampere: free_edges (4 n_free), b + W_all (16 nF), Dinv + e r/w (24 n_free)
    b_ampere = 28 * n_free + 16 * nF
    if n_hang > 0:
        b_ampere += 4 * (n_free + 1) + 4 * n_hang + 12 * nnz_pht
    if stepper.has_cpml:
        b_ampere += 8 * n_free + 32 * int(stepper._n_phi)

    b_lts = 0
    if stepper.has_lts:
        p_prefix = 1
        for lv in stepper.lts_levels[1:]:
            p = int(lv["p"])
            n_fl = int(lv["nF"])
            n_phil = int(lv["nPhi"])
            n_cl = int(lv["C_root"].size)
            nnz_fc = int(lv["fc_op"].val.size)
            nnz_kh = int(lv["kh_op"].val.size)
            b_face = 24 * n_phil + 12 * nnz_fc + 8 * (n_fl + n_cl)
            b_edge = 44 * n_fl + 8 * n_phil + 12 * nnz_kh
            b_lts += p_prefix * (p - 1) * (b_face + b_edge)
            p_prefix *= p

    return int(b_faraday + b_ampere + b_lts)


def _time_callable(step_fn, n_warmup=20, n_steps=200):
    """Time `n_steps` invocations of `step_fn(n_steps)` with `cp.cuda.Event`."""
    step_fn(n_warmup)
    cp.cuda.Device().synchronize()
    ev0 = cp.cuda.Event()
    ev1 = cp.cuda.Event()
    ev0.record()
    step_fn(n_steps)
    ev1.record()
    ev1.synchronize()
    return float(cp.cuda.get_elapsed_time(ev0, ev1)) / n_steps


def bench_row(kind, h_coarse, ratio, copy_bw_gb_s, n_steps=200):
    """Benchmark all GPU arms on a single `(kind, h_coarse, ratio)` configuration."""
    rng = np.random.default_rng(0)
    s_cpu = make_solver("big", kind, h_coarse, ratio)
    dt = float(s_cpu.dt)
    n_free = int(s_cpu.n_free)
    n_faces = int(s_cpu.n_faces)
    n_all_faces = int(s_cpu.mesh.n_faces)
    unknowns = n_free + n_faces
    e0 = rng.standard_normal(n_free)

    row = {
        "kind": kind,
        "h_coarse": float(h_coarse),
        "ratio": int(ratio),
        "h_fine": float(h_coarse / ratio),
        "n_free": n_free,
        "n_faces": n_faces,
        "n_all_faces": n_all_faces,
        "unknowns": unknowns,
        "dt": dt,
        "n_steps": int(n_steps),
    }

    # 1. SolverFIT3D baselines on uniform meshes (same grid and dt):
    #    CSR (per-step Python), fused (per-step Python), and fused (CUDA graph, chunk=64)
    if kind == "uniform":
        axes = solver_axes("big", "uniform", h_coarse, 1)
        for key, fused, use_graph in (
            ("fit3d_csr", False, False),
            ("fit3d_fused", True, False),
            ("fit3d_fused_graph", True, True),
        ):
            g = GridFIT3D(x=axes[0], y=axes[1], z=axes[2], verbose=0)
            sv = SolverFIT3D(
                g,
                bc_low=["pec"] * 3,
                bc_high=["pec"] * 3,
                verbose=0,
                dt=dt,
                cfln=0.95,
                use_gpu=True,
                fused=fused,
            )
            if use_graph:
                sv.one_step()
                loop = FusedLoop(sv, [], [], 64 + n_steps + 2, chunk=64)
                ms = _time_callable(
                    lambda k, loop=loop: loop.run(1, 1 + k), 64, n_steps
                )
                del loop
            else:
                ms = _time_callable(
                    lambda k, sv=sv: [sv.one_step() for _ in range(k)], 20, n_steps
                )
            row[f"{key}_ms"] = ms
            row[f"{key}_ns_per_dof"] = ms * 1e6 / unknowns
            row[f"{key}_time_T_s"] = (T_PHYS / float(sv.dt)) * ms * 1e-3
            del sv, g
            cp.get_default_memory_pool().free_all_blocks()

    # 2. Existing CuPy subgrid paths: csr and block
    for kernels in ("csr", "block"):
        key = f"subgrid_{kernels}"
        sg = make_solver(
            "big", kind, h_coarse, ratio, dt=dt, kernels=kernels, backend="cupy"
        )
        sg.e[:] = cp.asarray(e0)
        ms = _time_callable(lambda k, sg=sg: sg.run(k), 16, n_steps)
        row[f"{key}_ms"] = ms
        row[f"{key}_ns_per_dof"] = ms * 1e6 / unknowns
        row[f"{key}_time_T_s"] = (T_PHYS / dt) * ms * 1e-3
        del sg
        cp.get_default_memory_pool().free_all_blocks()

    # 3. Fused subgrid kernels: direct launches (graph_chunk=0) and CUDA graph (graph_chunk=64)
    for graph_chunk, key in ((0, "subgrid_fused_direct"), (64, "subgrid_fused_graph")):
        s_f = make_solver("big", kind, h_coarse, ratio, dt=dt, kernels="block")
        s_f.e[:] = e0
        fs = FusedSubgridStepper(s_f, graph_chunk=graph_chunk)
        warmup = 64 if graph_chunk > 0 else 20
        ms = _time_callable(fs.run, warmup, n_steps)
        row[f"{key}_ms"] = ms
        row[f"{key}_ns_per_dof"] = ms * 1e6 / unknowns
        row[f"{key}_time_T_s"] = (T_PHYS / dt) * ms * 1e-3
        if key == "subgrid_fused_graph":
            b_step = bytes_per_step_model(fs)
            roof_ms = (b_step * 1e-9) / copy_bw_gb_s * 1e3
            row["bytes_per_step"] = b_step
            row["bytes_per_unknown"] = float(b_step / unknowns)
            row["roofline_ms"] = float(roof_ms)
            row["kernels_per_step"] = int(fs.kernels_per_step())
        del fs, s_f
        cp.get_default_memory_pool().free_all_blocks()

    del s_cpu
    cp.get_default_memory_pool().free_all_blocks()

    row["speedup_vs_block"] = float(
        row["subgrid_block_ms"] / row["subgrid_fused_graph_ms"]
    )
    row["speedup_vs_csr"] = float(row["subgrid_csr_ms"] / row["subgrid_fused_graph_ms"])
    if kind == "uniform":
        row["ratio_direct_vs_fit3d_py"] = float(
            row["subgrid_fused_direct_ms"] / row["fit3d_fused_ms"]
        )
        row["ratio_graph_vs_fit3d_graph"] = float(
            row["subgrid_fused_graph_ms"] / row["fit3d_fused_graph_ms"]
        )

    tag = f"{kind}_H{h_coarse:.4f}_r{ratio}"
    extra = (
        f" fit3d_csr_ms={row['fit3d_csr_ms']:.4f}"
        f" fit3d_fused_ms={row['fit3d_fused_ms']:.4f}"
        f" fit3d_fused_graph_ms={row['fit3d_fused_graph_ms']:.4f}"
        f" ratio_direct_vs_fit3d_py={row['ratio_direct_vs_fit3d_py']:.2f}"
        f" ratio_graph_vs_fit3d_graph={row['ratio_graph_vs_fit3d_graph']:.2f}"
        if kind == "uniform"
        else ""
    )
    print(
        f"NUMBERS bench_fused: case={tag} unknowns={unknowns} "
        f"kernels_per_step={row['kernels_per_step']} "
        f"bytes_per_step={row['bytes_per_step']} "
        f"bytes_per_unknown={row['bytes_per_unknown']:.1f} "
        f"roofline_ms={row['roofline_ms']:.4f} "
        f"subgrid_csr_ms={row['subgrid_csr_ms']:.4f} "
        f"subgrid_block_ms={row['subgrid_block_ms']:.4f} "
        f"subgrid_fused_direct_ms={row['subgrid_fused_direct_ms']:.4f} "
        f"subgrid_fused_graph_ms={row['subgrid_fused_graph_ms']:.4f} "
        f"speedup_vs_block={row['speedup_vs_block']:.1f} "
        f"speedup_vs_csr={row['speedup_vs_csr']:.1f}"
        f"{extra}",
        flush=True,
    )
    return row


def main():
    ap = argparse.ArgumentParser(
        description="Benchmark fused GPU kernels for 3D subgridding."
    )
    ap.add_argument(
        "--quick", action="store_true", help="Run a small subset (< 2 min)."
    )
    ap.add_argument(
        "--out", required=True, help="Output directory for bench[_quick].json."
    )
    ap.add_argument(
        "--steps", type=int, default=200, help="Timed steps per arm (default 200)."
    )
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    fname = "bench_quick.json" if args.quick else "bench.json"
    out_path = os.path.join(args.out, fname)

    props = cp.cuda.runtime.getDeviceProperties(0)
    gpu_name = (
        props["name"].decode()
        if isinstance(props["name"], bytes)
        else str(props["name"])
    )
    copy_bw_gb_s, copy_bytes, copy_ms = measure_copy_bandwidth()
    print(
        f"NUMBERS bench_fused: gpu={gpu_name.replace(' ', '_')} "
        f"copy_bw_gb_s={copy_bw_gb_s:.2f} copy_bytes={copy_bytes} "
        f"copy_ms={copy_ms:.4f} graph_chunk=64 steps={args.steps}",
        flush=True,
    )

    if args.quick:
        uni_hs = [0.1, 0.05]
        sg_specs = [
            ("global", 0.1, 2),
            ("lts", 0.1, 2),
            ("lts", 0.1, 3),
            ("lts", 0.05, 2),
        ]
    else:
        uni_hs = [0.1, 0.05, 0.1 / 3, 0.025, 0.02, 0.05 / 3, 0.0125]
        sg_specs = [
            ("global", 0.1, 2),
            ("lts", 0.1, 2),
            ("global", 0.05, 2),
            ("lts", 0.05, 2),
            ("global", 0.1 / 3, 2),
            ("lts", 0.1 / 3, 2),
            ("global", 0.025, 2),
            ("lts", 0.025, 2),
            ("global", 0.1, 3),
            ("lts", 0.1, 3),
            ("global", 0.05, 3),
            ("lts", 0.05, 3),
            ("global", 0.1 / 3, 3),
            ("lts", 0.1 / 3, 3),
        ]

    t0 = time.time()
    rows = []
    for h in uni_hs:
        rows.append(bench_row("uniform", h, 1, copy_bw_gb_s, n_steps=args.steps))
    for kind, h, r in sg_specs:
        rows.append(bench_row(kind, h, r, copy_bw_gb_s, n_steps=args.steps))

    payload = {
        "meta": {
            "host": platform.node(),
            "gpu_name": gpu_name,
            "copy_bw_gb_s": copy_bw_gb_s,
            "copy_bytes": copy_bytes,
            "copy_ms": copy_ms,
            "quick": bool(args.quick),
            "n_steps": int(args.steps),
            "box_m": list(BOX),
            "t_phys_s": T_PHYS,
            "wall_s": float(time.time() - t0),
        },
        "rows": rows,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {out_path} ({payload['meta']['wall_s']:.1f} s)", flush=True)


if __name__ == "__main__":
    main()
